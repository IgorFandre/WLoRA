import torch, gc, os, sys, wandb, peft, json
import numpy as np
from transformers import (
    Trainer,
    HfArgumentParser,
    get_scheduler,
)

from utils_glue import glue_preprocess
sys.path.append(os.getcwd())
from src import (
    config,
    optimizers,
    utils
)
import warnings
warnings.filterwarnings("ignore")

def main():
    for i in range(torch.cuda.device_count()):
        print("We will use the GPU:", torch.cuda.get_device_name(i))
    parser = HfArgumentParser((
        config.ModelArguments,
        config.DataTrainingArguments,
        config.TrainingArguments
    ))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    utils.set_seed(training_args.seed)
    ################# Model, Tokenizer and Dataset Downloading #################
    (train_dataset, eval_dataset, test_dataset, datasets, data_collator, 
     compute_metrics, model, tokenizer) = glue_preprocess(data_args, 
                                                          training_args, 
                                                          model_args)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id
    ############################### PEFT Adapters ##############################
    all_params_before_peft, _ = utils.print_trainable_parameters(model, verbose=False)
    training_args.model_name = model_args.model_name_or_path         # for wandb
    peft_args = utils.get_peft_arguments(training_args)
    if peft_args is not None:
        model = peft.get_peft_model(model, peft_args)
    for name, param in model.named_parameters():
        if "attention.self" not in name and "output.dense" not in name and "intermediate.dence" not in name:
            param.requires_grad = False
    num_peft_adapters = utils.count_atapters(model, training_args.ft_strategy)

    #training_args.ft_strategy = "FatLoRA"
    training_args.label_names = ["labels"] # peft and compute_metrics() problem
    ######################### Optimizer and Scheduler ##########################
    if "tuned" in [training_args.learning_rate]: # [TODO] add more tuned params
        f_name = "./glue_experiment/tuned_params.json"
        with open(f_name) as f:
            tuned_params = json.load(f)
        lr = tuned_params[data_args.task_name][training_args.ft_strategy]["lr"]
        training_args.learning_rate = lr
    else:
        training_args.learning_rate = float(training_args.learning_rate)

    # Set optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_args.learning_rate,
        weight_decay=training_args.weight_decay
    )

    ############################### Wandb Saves ################################
    training_args.all_params, training_args.trainable_params = \
        utils.print_trainable_parameters(model)
    training_args.num_peft_adapters = num_peft_adapters
    training_args.peft_params = training_args.all_params - all_params_before_peft
    training_args.train_proportion = training_args.trainable_params / training_args.all_params * 100 
    training_args.peft_proportion = training_args.peft_params / training_args.all_params * 100 
    os.environ["WANDB_PROJECT"] = "SIMPLEX_WLORA"
    if training_args.ft_strategy in ["WeightLoRA", "RandLoRA"]:
        run_name = f"[{training_args.ft_strategy} k={training_args.k} r={training_args.lora_r}]"
    else:
        run_name = f"[{training_args.ft_strategy} r={training_args.lora_r}]"
    run_name += f" {data_args.task_name}, lr={training_args.learning_rate}"
    training_args.run_name = run_name
    training_args.output_dir = f"./glue_experiment/{training_args.output_dir}/{run_name}"
    os.environ["WANDB_TAGS"] = f"GLUE {data_args.task_name} NEW"
    if optimizer is not None:
        training_args.optimizer = optimizer.__class__.__name__
    else:
        training_args.optimizer = training_args.optim
    training_args.benchmark_name = data_args.dataset_name
    training_args.tsk_name = data_args.task_name
    ############################# Training #####################################
    print("$"*30, f" {run_name} ", "$"*30)

    ############## FAT STEP ##############
    training_args.max_steps = 10
    
    trainer=Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        data_collator=data_collator,
        optimizers=[optimizer, None] #
    ) #### WARNING с обычным адамом маленький lr, и w не успевают обучиться
    _ = trainer.train()
    
    w = []
    for name, param in model.named_parameters():
        if 'weight_lora_w' in name:
            w.append(param.data.item())

    w_0 = np.array(w)
    w_0 = w_0 - w_0.max()
    soft_w = np.exp(w_0)/sum(np.exp(w_0))
    
    lora_dict = dict()
    w_idx = 0
    for name, param in model.named_parameters():
        if 'weight_lora_A' in name:
            lora_dict[name] = param

        if 'weight_lora_B' in name:
            lora_dict[name] = param

        if 'weight_lora_w' in name: # Встречается последним в перечислении параметров
            param.data = torch.tensor(1., device=param.device, requires_grad=False)

            A_name = name.replace('weight_lora_w', 'weight_lora_A')
            B_name = name.replace('weight_lora_w', 'weight_lora_B')
            
            A = lora_dict[A_name]
            B = lora_dict[B_name]
            
            r_old = A.data.shape[1]
            r_new = int(num_peft_adapters * soft_w[w_idx] * r_old)

            print(f'{r_old}, {r_new}')
            if r_new > r_old:
                utils.upgrade_lora_AB(A, B, r_new)
                break
            elif r_new == 0:
                A.data = torch.zeros((A.data.shape[0], 1), requires_grad=False, device=A.data.device)
                B.data = torch.zeros((1, B.data.shape[1]), requires_grad=False, device=B.data.device)
                break
            elif r_new < r_old:
                print(f'begin: A = {A.shape}, B = {B.shape}')
                utils.downgrade_lora_AB(A, B, r_new)
                print(f'end: A = {A.shape}, B = {B.shape}')
                break
            
            torch.cuda.empty_cache()

            A.data = A.data * w[w_idx]
            
            # In previous verion:
            # state["step"] = 0
            # state["exp_avg"] = torch.zeros_like(p)
            # state["exp_avg_sq"] = torch.zeros_like(p) 
            # The problem:
            ### RuntimeError: The size of tensor a (4) must match the size of tensor b (3) at non-singleton dimension 1
            ### begin: A = torch.Size([768, 4]), B = torch.Size([4, 768])
            ### end: A = torch.Size([768, 3]), B = torch.Size([3, 768])
    
    #####################################

    training_args.max_steps = -1 # Default value
    trainer=Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        data_collator=data_collator,
        optimizers=[optimizer, None]
    )

    if training_args.do_train:
        train_result = trainer.train()
        train_metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        train_metrics["train_samples"] = min(max_train_samples, len(train_dataset))
        train_metrics["train_memory_gb"] = torch.cuda.max_memory_allocated() / 2**30
        train_metrics["train_runtime"] /= 60
        if training_args.ft_strategy in ["WeightLoRA", "RandLoRA", "FatLoRA"]:
            i = 0
            for name, param in model.named_parameters():
                if "weight_lora_w" in name:
                    if param.sum().item() > 0 and param.requires_grad:
                        i += 1
                        if training_args.model_name == "microsoft/deberta-v3-base":
                            tmp = name.split(".")
                            if "attention.self" in name:
                                layer_name = f"attn_{tmp[8].split('_')[0]}" 
                            elif "attention" in name:
                                layer_name = f"attn_{tmp[7]}"
                            else:
                                layer_name = tmp[6]
                            load_name = f"{layer_name}#{tmp[5]}"
                        else:
                            load_name = name
                        train_metrics[f"active_adapters_{i}"] = load_name

        trainer.save_model()

        trainer.log_metrics("train", train_metrics)
        trainer.save_metrics("train", train_metrics)
        trainer.save_state()

        if "wandb" in training_args.report_to:
            wandb.config.update(train_metrics, allow_val_change=True)
    ################################ Evaluation ################################
    if training_args.do_eval:
        # Loop to handle MNLI double evaluation (matched, mis-matched)
        tasks = [data_args.task_name]
        eval_datasets = [eval_dataset]
        if data_args.task_name == "mnli":
            tasks.append("mnli-mm")
            eval_datasets.append(datasets["validation_mismatched"])

        for eval_dataset, task in zip(eval_datasets, tasks):
            eval_metrics = trainer.evaluate(eval_dataset=eval_dataset)
            max_val_samples = data_args.max_val_samples if data_args.max_val_samples is not None else len(eval_dataset)
            eval_metrics["eval_samples"] = min(max_val_samples, len(eval_dataset))
            trainer.log_metrics("Eval_%s"%task, eval_metrics)
            trainer.save_metrics("Eval_%s"%task, eval_metrics)
            
        if "eval_runtime" in eval_metrics.keys():
            eval_metrics["eval_runtime"] /= 60
        if "wandb" in training_args.report_to:
            wandb.config.update(eval_metrics, allow_val_change=True)
    ################################# Testing ##################################
    if training_args.do_predict:
        # Loop to handle MNLI double evaluation (matched, mis-matched)
        tasks = [data_args.task_name]
        test_datasets = [eval_dataset]
        if data_args.task_name == "mnli":
            tasks.append("mnli-mm")
            test_datasets.append(datasets["validation_mismatched"])

        for test_dataset, task in zip(test_datasets, tasks):
            metrics = trainer.evaluate(test_dataset, metric_key_prefix="test")
            max_samples = data_args.max_val_samples if data_args.max_val_samples is not None else len(test_dataset)
            metrics["test_samples"] = min(max_samples, len(test_dataset))
            trainer.log_metrics("Test_%s"%task, metrics)
            trainer.save_metrics("Test_%s"%task, metrics)

    del trainer, model
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
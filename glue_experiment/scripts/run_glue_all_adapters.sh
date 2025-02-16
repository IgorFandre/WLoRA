for ft_strategy in LoRA LoKR LoHA ADALoRA
do
    CUDA_VISIBLE_DEVICES=1 python run_experiment.py \
        --dataset_name glue \
        --task_name stsb \
        --model_name_or_path microsoft/deberta-v3-base \
        --per_device_train_batch_size 32 \
        --per_device_eval_batch_size 32 \
        --gradient_accumulation_steps 6 \
        --learning_rate 8e-4 \
        --lr_scheduler_type linear \
        --warmup_steps 100 \
        --max_steps 512 \
        --eval_steps 64 \
        --save_steps 256 \
        --ft_strategy $ft_strategy \
        --lora_r 8 \
        --lora_alpha 32 \
        --lora_dropout 0.05 \
        --report_to wandb # none or wandb
done

CUDA_VISIBLE_DEVICES=1 python run_experiment.py \
    --dataset_name glue \
    --task_name stsb \
    --model_name_or_path microsoft/deberta-v3-base \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 32 \
    --gradient_accumulation_steps 6 \
    --learning_rate 8e-3 \
    --lr_scheduler_type linear \
    --warmup_steps 100 \
    --max_steps 512 \
    --eval_steps 64 \
    --save_steps 256 \
    --ft_strategy VERA \
    --lora_r 1024 \
    --lora_dropout 0.05 \
    --report_to wandb # none or wandb

CUDA_VISIBLE_DEVICES=1 python run_experiment.py \
    --dataset_name glue \
    --task_name stsb \
    --model_name_or_path microsoft/deberta-v3-base \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 32 \
    --gradient_accumulation_steps 6 \
    --learning_rate 8e-5 \
    --lr_scheduler_type linear \
    --warmup_steps 100 \
    --max_steps 512 \
    --eval_steps 64 \
    --save_steps 256 \
    --ft_strategy Full \
    --report_to wandb # none or wandb


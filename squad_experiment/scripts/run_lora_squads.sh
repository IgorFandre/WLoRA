for dataset_name in squad squad_v2
do
    for r in 8
    do
        for seed in 18 52 1917
        do
            clear
            CUDA_VISIBLE_DEVICES=0 python ./squad_experiment/run_squad.py \
            --dataset_name $dataset_name \
            --model_name_or_path microsoft/deberta-v3-base \
            --per_device_train_batch_size 32 \
            --per_device_eval_batch_size 32 \
            --gradient_accumulation_steps 6 \
            --learning_rate tuned \
            --lr_scheduler_type linear \
            --warmup_steps 100 \
            --max_steps 1024 \
            --eval_steps 64 \
            --save_steps 256 \
            --ft_strategy LoRA \
            --lora_r $r \
            --lora_alpha 32 \
            --lora_dropout 0.05 \
            --seed $seed \
            --report_to wandb # none or wandb
        done
    done
done
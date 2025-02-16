for dataset_name in squad squad_v2
# for dataset_name in squad_v2
do
    for r in 1 2 4 8
    do
        for seed in 18 52 1917
        do
            clear
            CUDA_VISIBLE_DEVICES=0 python ./squad_experiment/run_squad.py \
            --dataset_name $dataset_name \
            --model_name_or_path microsoft/deberta-v3-base \
            --per_device_train_batch_size 64 \
            --per_device_eval_batch_size 64 \
            --gradient_accumulation_steps 6 \
            --learning_rate 8e-5 \
            --learning_rate_w 8e0 \
            --lr_scheduler_type linear \
            --warmup_steps 100 \
            --max_steps 1024 \
            --eval_steps 64 \
            --save_steps 256 \
            --ft_strategy WeightLoRA \
            --lora_r $r \
            --lora_alpha 32 \
            --lora_dropout 0.05 \
            --k 20 \
            --seed $seed \
            --report_to wandb # none or wandb
        done
    done
done
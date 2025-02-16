for dataset_name in cnn_dailymail
do
    #for lr in 8e-5 5e-5 1e-4 3e-4 5e-4 8e-4 1e-3
    for lr in 8e-4
    do
        for seed in 18
        do
            CUDA_VISIBLE_DEVICES=3 python ./nlg_experiment/run_nlg.py \
            --dataset_name $dataset_name \
            --model_name_or_path facebook/bart-large \
            --dataset_config "3.0.0" \
            --per_device_train_batch_size 8 \
            --per_device_eval_batch_size 8  \
            --gradient_accumulation_steps 6 \
            --learning_rate $lr \
            --lr_scheduler_type linear \
            --warmup_steps 100 \
            --max_steps 512 \
            --eval_steps 128 \
            --save_steps 256 \
            --ft_strategy LoRA \
            --lora_r 8 \
            --lora_alpha 32 \
            --lora_dropout 0.05 \
            --seed $seed \
            --predict_with_generate true \
            --max_source_length 1024 \
            --max_target_length 160 \
            --val_max_target_length 1024 \
            --max_target_length 1024 \
            --max_val_samples 3000 \
            --report_to wandb # none or wandb
        done
    done
done
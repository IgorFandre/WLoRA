for r in 8 32 64 128 256
do
    for ((k=1;k<=36;k++));
    do
        CUDA_VISIBLE_DEVICES=0 python additional_experiments/run_memory.py \
            --dataset_name glue \
            --task_name mnli \
            --model_name_or_path microsoft/deberta-v3-base \
            --per_device_train_batch_size 64 \
            --per_device_eval_batch_size 64 \
            --gradient_accumulation_steps 6 \
            --learning_rate 8e-4 \
            --lr_scheduler_type cosine \
            --warmup_steps 50 \
            --max_steps 5 \
            --lora_r $r \
            --do_evaluate true \
            --seed 52 \
            --k $k \
            --report_to none
    done
done
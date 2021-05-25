OUTPUT_DIR="/data/dobby_ceph_ir/neutrali/venus_outputs/spell-pretrain_model-pho2-pretrain_bs-64_lr-5e-5_mxs-30000_seed-42"

python src/run_pretrain.py \
    --model_type pho2-pretrain \
    --output_dir $OUTPUT_DIR \
    --do_train --do_eval --do_predict  \
    --remove_unused_ckpts \
    --per_gpu_train_batch_size 64 \
    --gradient_accumulation_steps 2 \
    --per_gpu_eval_batch_size 50 \
    --learning_rate 5e-5 \
    --max_steps 30000 \
    --seed 42 \
    --warmup_steps 5000 \
    --eval_all_checkpoints \
    --overwrite_output_dir

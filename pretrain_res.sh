# num_fonts 3, use_traditional_font

python src/run_res_pretrain.py \
    --model_type res-pretrain \
    --output_dir /data/dobby_ceph_ir/hengdaxu/venus_outputs/pretrain_res/pretrain_res_seed42_epoch8_font3_fanti \
    --do_train --do_eval \
    --learning_rate 5e-5 \
    --warmup_steps 100 \
    --overwrite_output_dir \
    --remove_unused_ckpts \
    --seed 42 \
    --num_train_epochs 8 \
    --num_fonts 3 \
    --use_traditional_font

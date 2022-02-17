python3 -i train.py \
    --task csn \
    --sub_task csn_python \
    --lang python \
    --data_dir data/csn_sum/dataset/python \
    --cache_path data/csn_sum \
    --summary_dir checkpoint/csn_sum/summary \
    --res_dir checkpoint/csn_sum/result \
    --output_dir checkpoint/csn_sum/output \
    --do_train \
    --do_eval \
    --do_train \
    --do_eval_bleu \
    --num_train_epochs 5 \
    --learning_rate 5e-5 \
    --beam_size 10 \
    --patience 3 \
    --weight_decay 1e-5 \
    --warmup_steps 5000 \
    --max_source_length 256 \
    --max_target_length 512 \
    --batch_size 16 \
    --gradient_accumulation_steps 2 \
    --no_tqdm \
    --k 0

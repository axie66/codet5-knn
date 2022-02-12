# for i in 1 2 3 4 5
# do
python3 -i train_conala.py \
    --task conala \
    --lang python \
    --cache_path data/concode \
    --summary_dir checkpoint/summary \
    --res_dir checkpoint/result \
    --output_dir checkpoint/output \
    --data_dir foo \
    --do_train \
    --do_test \
    --do_eval \
    --num_train_epochs 5 \
    --batch_size 16 \
    --learning_rate 5e-5 \
    --beam_size 10 \
    --weight_decay 1e-5 \
    --max_target_length 512 \
    --k 0 \
    --seed 12761
# done

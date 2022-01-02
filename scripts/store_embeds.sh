python3 store_embeds.py \
    --pretrained_path checkpoint/finetuned_models_concode_codet5_base.bin \
    --dataset_path data/concode/train_all.pt \
    --data_name concode_train \
    --save_kv_pairs \
    --batch_size 32

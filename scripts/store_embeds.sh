python3 -i conala_store_embeds.py \
    --pretrained_path checkpoint/output/checkpoint-best-bleu/pytorch_model.bin \
    --data_type doc \
    --batch_size 32 \
    --mono_min_prob 0.1

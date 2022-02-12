#PRETRAINED_PATH=checkpoint/output/checkpoint-best-bleu/pytorch_model.bin
PRETRAINED_PATH=pretrained_weights/conala_codet5_base.bin

python3 store_embeds.py \
	--pretrained_path $PRETRAINED_PATH \
	--dataset concode \
	--use_train \
	--save_kv_pairs \
	--batch_size 8 \
	--cache_path data/concode \
	--data_dir data/concode/dataset/concode

#python3 -i conala_store_embeds.py \
#    --pretrained_path $PRETRAINED_PATH \
#    --use_doc --use_mined \
#    --batch_size 32 \
#    --mono_min_prob 0.1 \
#    --save_kv_pairs

#python3 -i conala_store_embeds.py \
#    --pretrained_path checkpoint/output/checkpoint-best-bleu/pytorch_model.bin \
#    --data_type train \
#    --batch_size 32 \
#    --mono_min_prob 0.1 \
#    --save_kv_pairs
#

#python3 -i conala_store_embeds.py \
#    --pretrained_path checkpoint/output/checkpoint-best-bleu/pytorch_model.bin \
#    --data_type doc \
#    --batch_size 32 \
#    --mono_min_prob 0.1 \
#    --save_kv_pairs

#python3 -i conala_store_embeds.py \
#    --pretrained_path checkpoint/output/checkpoint-best-bleu/pytorch_model.bin \
#    --data_type mined \
#    --batch_size 32 \
#    --mono_min_prob 0.1 \
#    --save_kv_pairs

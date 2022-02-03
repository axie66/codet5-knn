DSTORE_TYPE="doc-mined"
DSTORE_SIZE=1829724 # not using conala train data

python3 -i train_conala.py \
    --task conala \
    --lang python \
    --cache_path data/concode \
    --summary_dir checkpoint/summary \
    --res_dir checkpoint/result \
    --output_dir checkpoint/output \
    --model_name_or_path checkpoint/output/trained_model.bin \
    --data_dir foo \
    --do_train \
    --do_test \
    --do_eval \
    --num_train_epochs 10 \
    --batch_size 16 \
    --learning_rate 1e-4 \
    --beam_size 10 \
    --weight_decay 1e-5 \
    --max_target_length 128 \
    --k 32 \
    --knn_attn \
    --dstore-fp16 \
    --dstore-size ${DSTORE_SIZE} \
    --dstore-filename datastore/${DSTORE_TYPE} \
    --indexfile datastore/${DSTORE_TYPE}_knn.index \
    --seed 1234 \
    --move-dstore-to-mem \
    --no-load-keys \
    --faiss_gpu
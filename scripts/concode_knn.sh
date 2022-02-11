DSTORE_TYPE="concode"
DSTORE_SIZE=3535137

for lmbda in 0.05
do
    for k in 0
    do

    echo "Using k = ${k} and lmbda = ${lmbda}"

    python3 train_concode.py \
        --task concode \
        --lang java \
        --data_dir data/concode/dataset/concode \
        --cache_path data/concode \
        --summary_dir checkpoint/summary \
        --res_dir checkpoint/result \
        --output_dir checkpoint/output \
        --model_name_or_path pretrained_weights/concode_codet5_base.bin \
        --do_test \
        --batch_size 16 \
        --beam_size 10 \
        --wandb \
        --max_source_length 320 \
        --max_target_length 150 \
        --seed 1234 \
        --dstore-fp16 \
        --k ${k} \
        --probe 8 \
        --lmbda ${lmbda} \
        --knn_temp 10.0 \
        --dstore-size ${DSTORE_SIZE} \
        --dstore-filename datastore/${DSTORE_TYPE} \
        --indexfile datastore/${DSTORE_TYPE}_knn.index \
        --faiss_gpu \
        --move-dstore-to-mem \
        --no-load-keys
    done
done
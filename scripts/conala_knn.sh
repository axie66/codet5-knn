DSTORE_TYPE="mined"

if [ $DSTORE_TYPE = "mined" ]
then
    DSTORE_SIZE=1869575 # 1829724
elif [ $DSTORE_TYPE = "train" ]
then
    DSTORE_SIZE=39851
elif [ $DSTORE_TYPE = "doc" ]
then
    DSTORE_SIZE=195322
else
   echo "Error: Unknown DSTORE_TYPE '${DSTORE_TYPE}'"
   exit 1
fi

python3 -i train_conala.py \
    --task conala \
    --lang python \
    --cache_path data/concode \
    --summary_dir checkpoint/summary \
    --res_dir checkpoint/result \
    --output_dir checkpoint/output \
    --data_dir foo \
    --do_test \
    --batch_size 16 \
    --beam_size 10 \
    --wandb \
    --max_target_length 128 \
    --seed 1234 \
    --dstore-fp16 \
    --k 16 \
    --probe 8 \
    --lmbda 0.05 \
    --knn_temp 10.0 \
    --dstore-size ${DSTORE_SIZE} \
    --dstore-filename datastore/${DSTORE_TYPE} \
    --indexfile datastore/${DSTORE_TYPE}_knn.index
    # --no-load-keys

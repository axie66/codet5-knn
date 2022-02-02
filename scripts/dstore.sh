python3 build_dstore.py \
   --dstore_mmap datastore/doc-mined \
   --dstore_size 1829724 \
   --dimension 768 \
   --faiss_index datastore/doc-mined_knn.index \
   --num_keys_to_add_at_a_time 500000 \
   --starting_point 0 \
   --dstore_fp16 \
   --ncentroids 4096

# python3 build_dstore.py \
#    --dstore_mmap datastore/mined \
#    --dstore_size 1869575 \
#    --dimension 768 \
#    --faiss_index datastore/mined_knn.index \
#    --num_keys_to_add_at_a_time 500000 \
#    --starting_point 0 \
#    --dstore_fp16 \
#    --ncentroids 4096

# python3 build_dstore.py \
#     --dstore_mmap datastore/doc \
#     --dstore_size 195322 \
#     --dimension 768 \
#     --faiss_index datastore/doc_knn.index \
#     --num_keys_to_add_at_a_time 500000 \
#     --starting_point 0 \
#     --dstore_fp16 \
#     --ncentroids 4096

# python3 build_dstore.py \
#     --dstore_mmap datastore/train \
#     --dstore_size 39851 \
#     --dimension 768 \
#     --faiss_index datastore/train_knn.index \
#     --num_keys_to_add_at_a_time 500000 \
#     --starting_point 0 \
#     --dstore_fp16 \
#     --ncentroids 768

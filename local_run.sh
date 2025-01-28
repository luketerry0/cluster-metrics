torchrun \
--nnodes 1 \
--nproc_per_node 1 \
--rdzv_id $RANDOM \
--rdzv_backend c10d \
--rdzv_endpoint "127.0.0.1:64425" \
main.py 
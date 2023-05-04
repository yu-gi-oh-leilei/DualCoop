# !/bin/bash

# python \
# main_mlc.py --cfg config/dualcoop/voc2007_SGD.yaml \
# --output 'checkpoint/dualcoop/work18/pro05' \
# --prob 0.5 \
# --gpus 0 \
# --print-freq 10

# python \
# main_mlc.py --cfg config/baseline/voc2007.yaml \
# --output 'checkpoint/baseline/work18/pro05' \
# --prob 0.5 \
# --gpus 0 \
# --print-freq 10


# torchrun --nnodes=1 --nproc_per_node=1 --rdzv_id=100 --rdzv_backend=c10d \
# main_mlc.py --cfg config/dualcoop/voc2007_SGD.yaml \
# --output 'checkpoint/dualcoop/work8/pro05' \
# --prob 0.5 \
# --gpus 0 \
# --print-freq 10


# torchrun --nnodes=1 --nproc_per_node=2 --rdzv_id=100 --rdzv_backend=c10d \
# main_mlc.py --cfg checkpoint/Partial_baseline/test/config.yaml \
# --output 'checkpoint/Partial_baseline/test1' \
# --prob 0.5

torchrun --nnodes=1 --nproc_per_node=1 --rdzv_id=100 --rdzv_backend=c10d \
main_mlc.py --cfg config/dualcoop/voc2007_SGD.yaml \
--output 'checkpoint/dualcoop/voc2007/work1/pro05' \
--prob 0.5 \
--gpus 0 \
--print-freq 10
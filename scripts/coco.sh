# !/bin/bash

torchrun --nnodes=1 --nproc_per_node=2 --rdzv_id=100 --rdzv_backend=c10d \
main_mlc.py --cfg checkpoint/Partial_baseline/test/config.yaml \
--output 'checkpoint/dualcoop/coco2014/work1' \
--prob 0.5


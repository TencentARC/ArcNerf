#!/bin/bash

# one gpu, one machine, no launch
python train.py --gpu_ids 0 --name 'test_single_gpu' --configs './configs/default.yaml'



# one gpu, one machine, using launch
python -m torch.distributed.launch --nproc_per_node=1 --master_port=$RANDOM train.py \
--gpu_ids 0 --name 'test_single_gpu_launch' --configs './configs/default.yaml'

# multi-gpu, one machine, using launch
python -m torch.distributed.launch --nproc_per_node=2 --master_port=$RANDOM train.py \
--gpu_ids 0,1 --name 'test_multi_gpu_launch' --configs './configs/default.yaml'



# multi-gpu, multi-machine, using launch
python -m torch.distributed.launch $@ train.py \
--gpu_ids 0,1 --name 'test_multi_gpu_machine_launch' --configs './configs/default.yaml'
# need to specify `--nodes=2 --node_rank=0/1` when needed

#!/bin/bash

# one gpu, one machine
srun -p slurm_gpu --mpi=pmi2 --job-name single_gpu \
--gres=gpu:1 --ntasks=1 --ntasks-per-node=1 --nodes 1 --cpus-per-task=4 --kill-on-bad-exit=1 \
python train.py --gpu_ids 0 --name 'test_single_gpu_slurm' --configs './configs/default.yaml' --dist.slurm True


# multi-gpu, one machine
srun -p slurm_gpu --mpi=pmi2 --job-name multi_gpu \
--gres=gpu:4 --ntasks=4 --ntasks-per-node=4 --nodes 1 --cpus-per-task=4 --kill-on-bad-exit=1 \
python train.py --gpu_ids 0,1 --name 'test_multi_gpu_slurm' --configs './configs/default.yaml' --dist.slurm True


# multi-gpu, multi-machine, --ntasks = --ntask-per-node * --nodes, --gres=gpu is on each node
srun -p slurm_gpu --mpi=pmi2 --job-name multi_gpu \
--gres=gpu:4 --ntasks=4 -n 8 --ntasks-per-node=4 --nodes 2 --cpus-per-task=4 --kill-on-bad-exit=1 \
python train.py --gpu_ids 0,1,2,3 --name 'test_multi_gpu_machine_slurm' --configs './configs/default.yaml' --dist.slurm True

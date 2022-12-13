#!/bin/bash

# single cpu
python train.py --gpu_ids -1 --name 'test_cpu' --config './configs/default.yaml'

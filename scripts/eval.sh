#!/bin/bash

# on cpu
python eval_model.py --configs/eval.yaml --dir.eval_dir 'results/eval_xxx' --model_pt 'path_to_model'

# on gpu. Only allows one gpu
python eval_model.py --configs/eval.yaml --gpu_ids 0 --dir.eval_dir 'results/eval_xxx' --model_pt 'path_to_model'

#!/bin/bash

# on cpu
python evaluate.py --configs/eval.yaml --dir.eval_dir 'results/eval_xxx' --model_pt 'path_to_model'
python inference.py --configs/eval.yaml --dir.eval_dir 'results/eval_xxx' --model_pt 'path_to_model'

# on gpu. Only allows one gpu
python evaluate.py --configs/eval.yaml --gpu_ids 0 --dir.eval_dir 'results/eval_xxx' --model_pt 'path_to_model'
python inference.py --configs/eval.yaml --gpu_ids 0 --dir.eval_dir 'results/eval_xxx' --model_pt 'path_to_model'

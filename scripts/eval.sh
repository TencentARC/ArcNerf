#!/bin/bash

# on gpu only
python evaluate.py --configs/eval.yaml --gpu_ids 0 --dir.eval_dir 'results/eval_xxx' --model_pt 'path_to_model'
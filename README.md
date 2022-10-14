# ArcNerf


![nerf](assets/models/ngp.gif)

------------------------------------------------------------------------



------------------------------------------------------------------------
# Installation
Get the repo by `git clone https://github.com/TencentARC/ArcNerf --recursive`

- Install libs by `pip install -r requirements.txt`.
- Install the customized ops by `sh scripts/install_ops.sh`.
- Install tiny-cuda-nn modules by `sh scripts/install_tinycudann.sh`.

We test on env with:
- GPU: NVIDIA-A100 with CUDA 11.1 (Lower version may harm the `tinycudann` module).
- cmake: 3.21.3  (>=3.21)
- gcc: 8.3.1   (>=5.4)
- python: 3.8.5  (>=3.7)
- torch: 1.9.1

## Colmap
Colmap is used to estimate camera locations and sparse point cloud. It will let you run the algorithm on your own data.

Install under their [instruction](https://github.com/colmap/colmap).

------------------------------------------------------------------------
# Usage

## Data Preparation
- Download and prepare public datasets ref to [instruction](docs/datasets.md).
- If you use you own captured data, `scripts/data_process.sh` will help you extract the frames
and estimate the camera.

## Train
Train by `python train.py --configs configs/default.yaml --gpu_ids 0`.

- `--gpu_ids -1` will use `cpu`, which is good for you to debug the code in local IDE like pycharm line by line without a GPU device.
- for more details on the `config`, go to [default.yaml](configs/default.yaml) for more details.

## Evaluate
Eval by `python evaluate.py --configs configs/eval.yaml --gpu_ids 0`. You can set your target model by `--model_pt path/to/model`.

## Inference
Inference make customized rendering video and extract mesh output.
Run by `python inference.py --configs configs/eval.yaml --gpu_ids 0`. You can set your target model by `--model_pt path/to/model`.

## Notebook
Some notebooks are provided for you to understand what is happening for inference and how to use our visualizer.
Go to [notebook](notebooks) for more details.


------------------------------------------------------------------------
# What is special in this project?

In the recent few weeks, many frameworks working on common NeRF-based pipeline have been proposed:

- [NeRFStudio](https://github.com/nerfstudio-project/nerfstudio)
- [NeRF-Factory](https://github.com/kakaobrain/NeRF-Factory)
- [Wisp](https://github.com/NVIDIAGameWorks/kaolin-wisp)
- [JNeRF](https://github.com/Jittor/JNeRF)
- [XRNeRF](https://github.com/openxrlab/xrnerf)

All those amazing works are trying to bring those state-of-the-art NeRF-based methods together into
a complete, modular framework that is easy to change any of the components and conduct experiment quickly.

Toward the same goal, we are working on those fields could make this project helpful to the community:

- Highly modular design of pipeline:
  - Every field is seperated, and you can plug in any new developed module under the framework. Those fields
  can be easily controlled by the config files and modified without harming others.
  - We provide both sdf model, background model for modeling object and background as well, which are not commonly provided
  in other repo.


- Unified dataset and benchmark:
  - We separate the dataset based on official repo, and all methods are running under the same settings for fair comparison.
  - We also make unittests for the datasets and you are easy to check whether the setting on the data is correct.


- Many useful functionality are provided:
  - Mesh extraction on Density Model or SDF Model. (We are still working on incorporating better extraction functions)
  - Colmap preparation on your own capture data.
  - surface rendering on the sdf model
  - For other functionality on the trainer and logging, please ref [doc](docs/common_trainer.md).


- Docs and Code:
  - All the functions are with detailed docs on its usage, and the operation are comment with its tensor size,
  which makes you easy to understand the change to components.
  - We also provide many experiments [note](docs/expr.md) on our trails.


- Tests and Visual helpers:
  - We have developed an interactive visualizer to easily tests the correctness of our geometry function.
  - We have written a lots of unittest on the geometry and modelling functions. Take a lot, and you will be easy to understand how
  to use the visualizer for checking your own implementation.


We are still working on many other helpful functions, please ref [todo](docs/todolist.md) for more details. Bring issues to us
if you have any suggestions.


------------------------------------------------------------------------
## Datasets and Benchmarks

### Self-Capture data and Colmap


### Visual helper

![ray_pc](assets/datasets/ray_pc.gif)
![cam_pc](assets/datasets/cam_pc.gif)
![pts_pc](assets/datasets/pts_pc.gif)



------------------------------------------------------------------------
## Models


### full_model


### Base_3d_model

------------------------------------------------------------------------
## Geometry

------------------------------------------------------------------------
## Visualization

We make a offline interactive 3d visualizer in `plotly` backend. All the geometry components
in `numpy` tensor could be easily plugin the visualizer. It is compatible to torch-template 3d
projects and helpful for you to debug your implementation of the geometric functions.

We provide a [notebook](notebooks/draw_3d_examples.ipynb) showing the example of usage.
You can ref the [doc](docs/visual.md) for more details.

There is also another repo contain this visualizer.
Please go to [ArcVis](https://github.com/TencentARC/ArcVis) if you find it is helpful.

------------------------------------------------------------------------
## Code and Tests

We have made many unittests for checking the geometry function and models. See [doc](docs/tests.md) to know
how to test and get visual results.

We suggest to you make your own unittests if you are developing new algorithms to ensure correctness.

Comments in code are also helpful for you to learn how to use the function and the change of tensor size.

------------------------------------------------------------------------
## Trainer

We use our own training pipeline, which provides many customized functionality.
It is modular and easy to add/modify and part of the training pipeline.

We have another repo [common_trainer](https://github.com/TencentARC/common_trainer).
Or you can ref the [doc](docs/common_trainer.md) for more information.

------------------------------------------------------------------------
# License

TODO

------------------------------------------------------------------------
# Acknowledgements
Please see [Citation](docs/citation.md). Thanks to those amazing projects.

If you find this project useful, please consider citing:
```
@misc{arcnerf,
  author={Yue Luo, Yan-Pei Cao},
  title={arcnerf: nerf-based object/scene rendering and extraction framework },
  url={https://github.com/TencentARC/arcnerf/},
  year={2022},
```

You can contact the author by `leoyluo@tencent.com` if you need any help.

# A pytorch template for deep learning project
An easy-to-use template for pytorch dl projects.

------------------------------------------------------------------------
## Start a new proj
- Use `python start_new_proj.py --proj_name xx --proj_loc /path/to/proj_parent_dir` to extent to a new project.
What you need to implement are the data, model, loss, metric, progress_img_saver.
All other func for training and evaluation have been provided.
- When you start a new proj with proj_name, the custom lib will be renamed by your proj_name. Recommend to use
Camel-Case like (ProjName).

------------------------------------------------------------------------
## Installation
- Install required lib by `pip install -r requirements`. Major lib are: torch, numpy, loguru, tensorboard, pyyaml
- `pre-commit install` to install pre-commit for formatting. `pre-commit run --all-files` for checking all files.

------------------------------------------------------------------------
## Main Function
Use `python train.py --config configs/default.yaml` to start training.
All params should be referred to `configs/default.yaml`

------------------------------------------------------------------------
## CPU training
- Setting `--gpu_ids -1` will only use cpu, good for debugging. Refer `scripts/cpu.sh` for more detail.

## GPU and Multi process
- Use launch: You can refer to `script/gpu.sh` for training on gpu.
Single/Multi-gpu with local machine and distributed machines are allowed.
- Use slurm: You can refer to `script/slurm.sh` for training on gpu using `slurm`.
Single/Multi-gpu with local machine and distributed machines are allowed.

`@master_only` in all functions allows only the `rank=0` node performing func.

------------------------------------------------------------------------
## Config
- Use yaml to save configs. Mainly saved at `configs/`. If you want to set or update
by argument, you can directly add `--arg value` during input.

- All arguments in yaml are in levels, and input arguments should be `--level1.level2...`

------------------------------------------------------------------------
## Logging
- We use `loguru` to save and show the log. Only `rank=0` process shows the log. You can `add_log` and set msg_level

------------------------------------------------------------------------
## Resume training
- You can set `--resume` as the checkpoint_path, or the checkpoint folder which will load the `lastest.pt.tar`.
But this only reads the model, you have to set `--configs xxx` as the configs in the existing expr folder.

- In `resume` mode, if you set `progress.start_epoch` as `-1`. It will resume training.

- If `progress.start_epoch` is `0`, it will load the weight and fine-tune from epoch 0. You should set
a different expr name like `xxx_finetune` for separation.

------------------------------------------------------------------------
## Reproduce an old experiment
- All updated configs will be saved in the experiment. You just need to run `job.sh` in the exp to reproduce result.

- The script is for starting cpu training. You need to modify the `job.sh` to use gpu.

------------------------------------------------------------------------
## Model
- You can add your model at `custom.models` with `xxx_model.py`.

- Add `@MODEL_REGISTRY.register()` to the class for registration.

- Some backbones/components are provided in `common.models`.

------------------------------------------------------------------------
## Dataset
- `dir.data_dir` in config is the main data_dir for all dataset. Should not specify it for any single dataset.
You should modify you `custom.xx_dataset.py` to make the address specified for you dataset.

- You can add your dataset at `custom.datasets` with `xxx_dataset.py`.

- Add `@DATASET_REGISTRY.register()` to the class for registration.

To set dataset used in train/val/eval, set
```
dataset
    train:
        type: xxDataset
        augmentation:
            xxx:
    val:
    eval:
```
Missing val/eval will not do validation and eval during training.

## Data Transforms
- You can modify the function `custom.dataset.transform.get_transforms` for choosing data transformation.

- Some basic function are provided in `common.dataset.transform.augmentation`.

------------------------------------------------------------------------
## Loss
- You can add your loss at `custom.loss` with `xxx_loss.py`.

- Add `@LOSS_REGISTRY.register()` to the class for registration.

To set loss
```
loss:
    loss1:
        weight: 1.0
        other: xxx
        augmentation:
    loss2:
        weight: 2.0
```
- Weights will be combined in loss_factory in `custom.loss.__init__`, you don't need to multiply weight in
each implementation.

- When implementing metric, you have to put `inputs` to the `output` device. Refer to `custom.loss.img_loss` for example.


The resulting loss dict will be:
```
loss:
    names: [loss1, loss2, ...]
    loss1: xx.xx
    loss2: xx.xx
    ...
    sum: xx.xx
```

------------------------------------------------------------------------
## Metric
- Similar to Loss to calculate all metrics in once. But you don't need to set weights here, and no 'sum' is calculated.
-
- Add `@METRIC_REGISTRY.register()` to the class for registration.

- When implementing metric, you have to put `inputs` to the `output` device. Refer to `custom.metric.custom_metric` for example.

- The resulting metric dict will be:
```
metric:
    names: [metric1, metric2, ...]
    metric1: xx.xx
    metric2: xx.xx
    ...
```

------------------------------------------------------------------------
## Grad clip
- Support grad on the whole model by `clip_gradients`.
You can set `clip_warm` as positive number in order to use `clip_gradients_warmup` after warmup period.

------------------------------------------------------------------------
## Valid
- Validation will be performed on `val` dataset every `progress.epoch_val` epoch. Monitor will record result like loss, imgs.

- You can specify the valid cfgs in `dataset.val` to change the dataset details.

- If `progress.save_progress_val` is `True`, will save `progress.max_samples_val` result into `experiments/expr_name/progress/val`.

------------------------------------------------------------------------
## Eval
- Evaluation will be performed on `eval` dataset every `progress.epoch_eval` dataset.
All result will be locally recorded in `experiments/expr_name/eval` for each epoch.
But generally you should not make it in training progress. Local evaluation is better to avoid over-fitting.

- You can specify the valid cfgs in `dataset.eval` to change the dataset details.

- Metric will be needed for quantitative evaluation.

- If `progress.init_eval` is `True`, will evaluate with init model or resume model.
### Local Evaluation
- If you want to evaluate on a trained model, you can use `python evaluate.py` and set `--configs configs/eval.yaml` and
`--model_pt /path/to/model` for evaluation. Result will be written to `--dir.eval_dir results/eval_sample`.

- `eval.yaml` should contain param for `--dataset.eval`, `--model`, `--metric`.

------------------------------------------------------------------------
## Tests
- Tests for `common` class and `custom` are in `tests`. You should implement your tests for `custom` class when needed.

- We use unittest. You can run
  - `python -m unittest test_file` on tests in the whole file.
  - `python -m unittest discover test_dir` on tests in the whole directory.
  - `python -m unitttest test_dir.test_file.test_method` on test for single func.

------------------------------------------------------------------------
## Monitor and Progress saver
- A tensorboard monitor will be used during training to record train/val loss, vals, images, etc.

- All result in progress will be saved in `experiments/expr_name/event`. Use `tensorboard --logdir=experiments/expr_name/event` to check.

- At the same time, if you set `progress.local_progress` as True, imgs will be written to `experiments/expr_name/progress`.

- Change `render_progress_img` in `custom_trainer` for different visual results.

------------------------------------------------------------------------
## CUDA extension
We provide simple samples of CUDA extensions for simple add_matrix function, and a python wrapper
to use it like a `torch.nn.Module`.
More detail please see [official doc](https://pytorch.org/tutorials/advanced/cpp_extension.html).

Install it by getting into `custom/ops` and run `python setup.py install`. Or run `sh ./scripts/install_ops.sh`.

Run it by `python custom/ops/add_matrix.py` or
run tests by `python -m unittest tests/tests_custom/tests_ops/tests_ops.py`.

### Develop new ops
You need to have a new folder in `custom/ops/` to include the source cpp-wrapper and cuda implementation.

A python wrapper is suggested to put under `custom/ops/func.py` to use the func for usage.

### __global__, __device__, __host__: keywords
- `__global__`: call by cpu, run on gpu. Function must be `void`.
- `__device__`: call by gpu, run on gpu
- `__host__`: call by cpu, run on cpu
- `__host__ __device__`: both cpu and gpu
- `__global__ __host__` is not allow.

### grid-block-thread
`grid - block - thread` is the level structure of GPU computation unit.
- index = blockIdx.x * blockDim.x + threadIdx.x = the thread id in a grid
- stride = blockDim.x = total num of thread in a block. Commonly a block can be used to handle one batch.
- stride = blockDim.x * gridDim.x  = total num of thread in a grid
  - use this is called `grid-stride loop`
#### 2d and 1d
- 2d/1d grid/block are all supported based on your input tensor shape.
- Ref to [doc1](http://www.mathcs.emory.edu/~cheung/Courses/355/Syllabus/94-CUDA/2D-grids.html) and [doc2](https://blog.csdn.net/canhui_wang/article/details/51730264) for detail.

### PackedAccessor
To put a tensor into cuda kernel, it uses
`
    AT_DISPATCH_FLOATING_TYPES(A.scalar_type(), "sample_cuda",  // this will switch actual scalar type
    ([&] {
        kernel_func<scalar_t><<<blocks, threads>>>(
            A.data_ptr<scalar_t>(), B.data_ptr<scalar_t>(),
        );
    }));
`
If you use `A.data_ptr<scalar_t>()` to send the pointer, it will be hard to access the elements in kernel func.

You can instead use `PackedAccessor`, which is like
`torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>()` to allow easier access.

### cal_grad in forward
In some case, it is helpful to store by-product for backward grad calculation. But in pure inference mode, it is not
good to do such calculation during forward pass. It is helpful to pass an indicator in customized forward pass.

This indicator should be [`any(input.requires_grad)` and `torch.is_grad_enabled()`] to check
whether any input requires_grad and whether it is in the no_grad context. In the `.cu` kernel, you should have the
grad calculation by yourself.

------------------------------------------------------------------------
## More to do:
- inference, demo
- onnx or other implementation
- deploy and web server
- online project homepage
- colab
- setup.py

------------------------------------------------------------------------
## Acknowledge
This project template refers to:
- https://github.com/xinntao/ProjectTemplate-Python
- https://github.com/ventusff/neurecon#volume-rendering--3d-implicit-surface
- https://github.com/kwea123/pytorch_cppcuda_practice

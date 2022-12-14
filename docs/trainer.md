# ArcnerfTrainer
We use this for general nerf model training. (NeRF/Neus and their improved version).

## Train data process scheduler:
The scheduler handles train data only. It can customize the ray selection every time when all rays have been trained.
Add `scheduler` in `dataset.train` for specification.

A `Pipeline` class is in the trainer dir to process all the data, and sample data_batch for training.

- precrop: precrop and keep only the center rays.
  - max_epoch: Only crop when epoch < max_epoch. By default None(crop all the time).
  Set positive num to use it in the init rounds.
  - ratio: Ratio to keep in each dim.
  - If precrop is used, you are not allowed to use `shuffle` in data augmentation.

- ray_sample:
  - mode: `['full', 'random']`, full takes all rays, random is sample with replacement
    - 'full' mode takes all the rays for training. It will shuffle every time all rays have been processed
    - 'random': random sample rays in batch randomly with replacement. Some rays may not be sampled in this mode.
  - cross_view: used in both mode. If True, each sample takes rays from different image. Else on in one image.

- bkg_color:
  - used to blend any bkg_color to the training batch. Only when 'mask' exists.
  - color: If you use `random`, will use random bkg color for background rays. Otherwise, you can use (1, 1, 1) or other
  to set the rgb value.

- dynamic_batchsize:
  - Use for fg model to dynamically adjust num of rays
  - update_epoch: set up update num of rays every update epoch.
  - max_batch_size: By default set to 32768. Use it to forbid extreme large batch size

# get_progress and save visual 3D result
If you open `get_progress` in `debug`, and set `--progress.save_progress(val) True`, the trainer will save 3d sample points
with object bound in `expr_dir/progress/train(val)`.
- But `get_progress` accumulate lots of points, which makes CUDA memory out in some cases, be careful to use.

![pruning_pc](../assets/models/pruning_pc.gif)
![pruning_pts](../assets/models/pruning_pts.gif)

# Eval/Inference
Refer to [datasets](datasets.md) for more details.

# Web view
Thanks to [nerfstudio](https://github.com/nerfstudio-project/nerfstudio), we directly adopt their viewer in this project.
You can add the config in [viewer](../configs/viewer.yaml) in any of the training cfg, and view the result on web browser.

![viewer](../assets/viewer/ns_viewer.gif)

------------------------------------------------------------------------
## Common_Trainer
The details of how to use the trainer please ref to [common_trainer](common_trainer.md) and our project
[common_trainer](https://github.com/TencentARC/common_trainer).

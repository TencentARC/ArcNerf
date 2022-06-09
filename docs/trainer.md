# ArcnerfTrainer
We use this for general nerf model training. (NeRF/Neus and their improved version).

## Train data process scheduler:
The scheduler handles train data only. It can customize the ray selection every time when all rays have been trained.
Add `scheduler` in `data.train` for specification.

- precrop: precrop and keep only the center rays.
  - max_epoch: Only crop when epoch < max_shuffle. By default -1. Set positive num to use it in the init rounds.
  - ratio: Ratio to keep in each dim.
  - If precrop is used, you are not allowed to use `shuffle` in data augmentation.

- ray_sample:
  - mode: `['full', 'random']`, full takes all rays, random is sample with replacement
    - 'full' mode takes all the rays for training. If the image is with large white background, may not converge well
    - 'random': random sample rays in batch randomly with replacement. Some rays may not be sampled in this mode.
  - cross_view: used in both mode. If True, each sample takes rays from different image. Else on in one image.
  - NOTICE:
    - For white_bkg image like nerf_synthetic(lego, etc), we suggest use `random_nocrossview`, which will takes
    rays randomly from one image each time. This reduces the training on bkg rays.
    - For non white_bkg image like capture data, we suggest use `full_crossview` to optimize the model in different view.
    random will cause loss of rays.

- sample_loss: This use loss to do importance sampling during training. Which will keep all rays with large loss,
and randomly select rays with smaller loss.
    - min_sample: only use when n_shuffle >= min_sample. Set 0 to use in the very beginning.
    - ImgLoss/MaskLoss:  What loss to used.
      - keys: key of output component
      - loss_type: type of loss, MSE/L1/etc...
      - do_mean: Must be False to keep dim.
    - sampling: Controls how to do the importance sampling
      - threshold: min error to filter the large loss. depends on the actual loss type(mse, l1, etc)
      - random_ratio: num of ratio that select rays with small error.

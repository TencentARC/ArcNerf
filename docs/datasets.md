# base_3d_dataset
Base class for all 3d dataset. Contains image/mask(optional)/camera.
Support precache_ray/norm_cam_pose/rescale_image_pose/get_item in a uniform way.
## base_3d_pc_dataset
Based on `base_3d_dataset`, it provides functions mainly on point cloud adjustment.
Point cloud are in world coordinate.

# Some configs in data processing:
- img_scale: Resize image by this scale. `>1` means larger image.
Will change intrinsic for actual re-projection as well, but not change extrinsic.
- scale_radius: Rescale all camera pose such that cameras are roughly align on the surface of sphere with such radius.
Will not touch intrinsic. If point cloud exists, rescale them by same factor to keep consistency.
- precache: If True, will precache all the rays for all pixels at once.
- pc_radius(base_3d_pc_dataset): Remove point cloud that are outside such absolute radius.
Done before camera `scale_radius`.
## Augmentation:
- N_rays: Sample `N_rays` instead of using all, good for training.
## rgb and mask
- All color are in `rgb` order and normed by `255` into `0~1` range.
- All masks should be binary masks with `{0,1}` values.
## point cloud
- pts: `(n_pts, 3)` in world coordinate, `xyz` order
- color: `(n_pts, 3)`, `rgb` order, should be normed into `0~1` range.
- vis: `(n_cam, n_pts)`, visibility of each point in each cam. `{0,1}` values.
- pts is required, color/vis is optional.

## Capture
This class provides dataset from your capture data.
You need to run colmap to extract corresponding poses and train.
### Video_to_Image
If you capture video, you can follow `scripts/data_process.sh` to use `extract_video` to
extract video into images. Will write data to `cfgs.dir.data_dir/Capture/scene_name`
- video_path: actual video path
- scene_name: specify the scene name. Image will be written to `cfgs.dir.data_dir/Capture/scene_name/images`.
- video_downsample: downsample video frames by such factor. By default `15`.
- image_downsample: downsample each frame by such factor. By default `4`.
### Colmap Run_poses
You should install [colmap](https://colmap.github.io/) by yourself. We provide python script for processing.
you can follow `scripts/data_process.sh` to use `run_poses` to get colmap with poses and dense reconstruction.
Will write data to `cfgs.dir.data_dir/Capture/scene_name`
- match_type: `sequential_matcher` is good for sequential ordered images.
              `exhaustive_matcher` is good for random ordered image.
- dense_reconstruct: If true, run dense_reconstruct and get dense point cloud and mesh.
### Mask generation
- TODO: We may add it in the future.
### Dataset
Use `Capture` class for this dataset. It is specified by scene_name.
- scene_name: scene_name that  is the folder name under `Capture`.

## DTU
Specified by scan_id, read image/mask/camera.
- scan_id: int num for item selection.


# Train/Val/Test

## Train
Use all images for training. Same resolution as required


## Val
Use all images for training, downsampled by 2/4 depends on shape.

Each valid epoch just input one image for rendering


## Test
Use three closest camera(to avg_cam) for metric evaluation, use same resolution, and use a custom cam path with xx videos for rendering video

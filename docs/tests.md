# Usage
For any test, if it's `unittest` class, you can run
- `python -m unittest tests.xxx.test_file.py` or
- `python -m unittest discover tests.xxx.test_dir`

to auto run all unittest for checking.

Test a single func, use
- `python -m unittest tests.xxx.test_file.TestDict.tests_method`.

Some tests do not use check but give visual results in `tests/xxx/results`

We set default cfg file path at `__init__.py`. You can change it and tests will read the config from it.

Many of the tests on `geometry` or `model` will save visible results in the result dir.

------------------------------------------------------------------------
# tests_common
Tests for common class. Directly obtained from `common_trainer` project.

------------------------------------------------------------------------
# tests_arcnerf
Tests for all nerf/3d related functions.


## tests_datasets
Tests for datasets. Including showing cameras, avg cams and test cam trajectory.
### tests_any_datasets
It will read datasets from `default.yaml`, save related visual results into results folder.
Change the config file in `tests_datasets.__init__`.

------------------------------------------------------------------------
## tests_geometry
Test geometry function.
### tests_mesh
Test mesh function, like cal normals, getting face centers.
### tests_point_cloud
Test mesh function, like point cloud io.
### tests_sphere
Test the sphere representation and circle/spiral line creation
### tests_poses
Test the poses creation and modification functions.
### tests_projection
Test projection functions. Make pixels and transform them to points, then project points
back to pixels. Check the difference.
### tests_transformation
Test function in transform, like normalization, rotation
### tests_triangle
Test function in triangle, like getting norm, getting circumcircle, etc.
### tests_ray
Test function in ray, like getting sphere-ray-intersection, point-ray distance, etc.
### tests_volume
Test function in volume, like volume pts/line/faces generation and visualization,
ray-volume intersection with coarse structure, etc.

------------------------------------------------------------------------
## tests_render
### tests_camera
Test camera creation, get_rays and project rays into pixel, visualize ray and ray points
Also check when we rescale camera and image, whether rays get projected correctly.
### tests_ray_helper
Test func for ray sampling, resample cdf/pdf, etc.

------------------------------------------------------------------------
## tests_models
Test all the models and components for correctness and time.
### tests_base_modules
#### tests_activation
Test activation functions.
#### tests_encoding
Test all kinds of encoder. Some encoders have different backend(`torch`/`cuda`/`tcnn`), will check correctness.
#### tests_encoding_benchmark
Test encoders speed and time.
#### tests_linear
Test linear layers
#### tests_linear_network
Test implicit/radiance function with encoder and linear network.
#### tests_tcnn_fusedmlp_network
Test implicit/radiance function with encoder and tcnn_fusemlp network. Much faster that torch linear.
### tests_benchmark
Test all the model and get the speed/time.
### tests_fg_model
Test fg model for ray sampling and forward.
### tests_bkg_model
Test each bkg model like nerf++.
### tests_nerf
Test NeRF model.
### tests_mipnerf
Test MipNeRF model.
### tests_nerfpp
Test NeRF model with nerf++ as bkg model.
### tests_neus
Test Neus model and its sdf_to_alpha method with sampling.
### tests_volsdf
Test VolSDF model and its sdf_to_sigma method with sampling.

------------------------------------------------------------------------
### tests_ops
Tests of the customized operations in CUDA extension.

------------------------------------------------------------------------
### tests_loss
Tests for loss functions. Including geo_loss, img_loss, mask_loss.

------------------------------------------------------------------------
### tests_metric
Tests for eval metric functions. Including img_metric

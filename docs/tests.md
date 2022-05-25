# Usage
For any test, if it's `unittest` class, you can run
`python -m unittest tests.xxx.test_file.py` or
`python -m unittest discover tests.xxx.test_dir`, to auto run
all unittest for checking.
Test a single func, use `python -m unittest tests.xxx.test_file.TestDict.tests_method`.

For some tests that are not `unittest`, they are python by
`python test_file.py`. They generally provide visual results.

We set default cfg file path at `__init__.py`. You can change it and tests will read the config from it.

## tests_common
Tests for common class. Directly obtained from `common_trainer` project.

## tests_arcnerf
Tests for all nerf/3d related functions.

### tests_datasets
Tests for datasets. Including showing cameras, avg cams and test cam trajectory.
#### tests_any_datasets
This tests is not `unittest`. It will read datasets from `default.yaml`,
save related visual results into results folder.
Change the config file in `tests_datasets.__init__`.

### tests_geometry
Test geometry function.
#### tests_mesh
Test mesh function, like cal normals, getting face centers.
#### tests_point_cloud
Test mesh function, like point cloud io.
#### tests_sphere
Test the sphere representation and circle/spiral line creation
#### tests_poses
Test the poses creation and modification functions.
#### tests_projection
Test projection functions. Make pixels and transform them to points, then project points
back to pixels. Check the difference.
#### tests_transformation
Test function in transform, like normalization, rotation
#### tests_triangle
Test function in triangle, like getting norm, getting circumcircle, etc.
#### tests_ray
Test function in ray, like getting sphere-ray-intersection, point-ray distance, etc.
#### tests_volume
Test function in volume, like volume pts/line/faces generation and visualization.


### tests_render
#### tests_camera
Test camera creation, get_rays and project rays into pixel, visualize ray and ray points
Also check when we rescale camera and image, whether rays get projected correctly.
#### tests_ray_helper
Test func for ray sampling, resample cdf/pdf, etc.

### tests_models
#### tests_base_modules
Test embedder, implicit function, radiance function, with correct input/output.
#### tests_bkg_model
Test each bkg model like nerf++.
#### tests_nerf
Test NeRF model.
#### tests_nerfpp
Test NeRF model with nerf++ as bkg model.
#### tests_neus
Test Neus model and its sdf_to_alpha method with sampling.
#### tests_volsdf
Test VolSDF model and its sdf_to_sigma method with sampling.
#### tests_volnet
Test the custom dense volnet method, which contains an explict volume and get values for pts
using trilinear interpolation.


### tests_loss
Tests for loss functions. Including geo_loss, img_loss, mask_loss.

### tests_metric
Tests for eval metric functions. Including img_metric

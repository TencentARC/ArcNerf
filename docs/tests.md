# Usage
For any test, if it's `unittest` class, you can run
`python -m unittest tests/xxx/test_file.py` or
`python -m unittest discover tests/xxx/test_dir`, to auto run
all unittest for checking.

For some tests that are not `unittest`, they are python by
`python test_file.py`. They generally provide visual results.

## tests_common
Tests for common class. Directly obtained from `common_trainer` project.

## tests_arcnerf
Tests for all nerf/3d related functions.

### tests_datasets
Tests for datasets. Including showing cameras, avg cams and test cam trajectory.
#### tests_any_datasets
This tests is not `unittest`. It will read datasets from `default.yaml`,
save related visual results into results folder.

### tests_geometry
Test geometry function.
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

### tests_render
#### tests_camera
Test camera creation, get_rays and project rays into pixel, visualize ray and ray points
Also check when we rescale camera and image, whether rays get projected correctly.


### tests_models
#### tests_base_modules
Test embedder, implicit function, radiance function, with correct input/output.

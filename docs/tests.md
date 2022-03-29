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
Tests for datasets.
#### tests_any_datasets
This tests is not `unittest`. It will read datasets from `default.yaml`,
save related visual results into results folder.

### tests_geometry
Test geometry function.
#### tests_projection
Test projection functions. Make pixels and transform to points, project them
back to pixels. Check the difference.
#### tests_transformation
Test function in transform

### tests_render

### tests_visual
Tests for visualization functions
#### tests_vis_camera
Tests vis_camera function. Which draws cam position given extrinsics.

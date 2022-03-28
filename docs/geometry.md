# points/pixel transformation
- To transform xyz points with pixel index
- xyz_world -> xyz_cam -> pixel -> xyz_cam -> xyz_world
  - xyz_world -> xyz_cam: w2c
  - xyz_cam -> pixel: intrinsic
  - pixel -> xyz_cam: intrinsic
  - xyz_cam -> xyz_world: c2w
  - c2w = invert(w2c)

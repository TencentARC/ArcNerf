# Geometry
We implement the geometry class and operations in torch. Some CUDA Kernel is optional for acceleration.

------------------------------------------------------------------------
# projection
Function for cam projection from 3d points into image pixels
- `xyz_world` -> `xyz_cam` -> `pixel` -> `xyz_cam` -> `xyz_world`
- `xyz_world` -> `xyz_cam`: w2c
- `xyz_cam` -> `pixel`: intrinsic
- `pixel` -> `xyz_cam`: intrinsic
- `xyz_cam` -> `xyz_world`: c2w
- Distortion is allowed in `world->pixel`

------------------------------------------------------------------------
# transformation
Provide functions for geometrical transformation, including cam pose, vec, points
- normalize: norm vec
- rotation: rotate a matrix by R
- axis/rot representation interchange

------------------------------------------------------------------------
# poses
Functions for create/modify cam poses
- invert pose: `c2w` <-> `w2c` transfer
- look_at/view_matrix: generate `c2w`
- average/center poses: get avg pose and recenter all poses
- get cam on sphere by different mode

![cam_spiral](../assets/geometry/cam_spiral.gif)
![camera_circle](../assets/geometry/camera_circle.gif)


------------------------------------------------------------------------
# sphere
Many function about sphere is provided, including:
- uv-xyz transform
- get sphere line/surface
- get spiral line
- get swing line

It provides help for camera path creation.

Any point on a unit sphere with `(0,0,0)` origin can be represented by `(u, v)`,
where `u` in `(0, 2pi)`, `v` in `(0, pi)`
- x = cos(u) * sin(v)
- y = cos(v)
- z = sin(u) * sin(v)

![sphere_line](../assets/geometry/sphere_line.png)
![sphere_ray](../assets/geometry/sphere_ray.gif)

------------------------------------------------------------------------
# ray
Functions for ray point. ray is `(rays_o, rays_d)`, rays_d is always assumed to be normalized.
- get_ray_point_by_zvals: get the real point on ray using `rays_o/rays_d/zvals`.
- closest_point_on_ray: find the closest point on ray to an existing point. zvals can not be negative.
- closest_point_to_two_rays: two rays and their closest pts pair with distance. All case applied(parallel, zvals<0)
- closest_distance_of_two_rays: distance of two ray. But need to assume rays are pointing inward.
- closest_point_to_rays: a point close to all rays. Good for cam view centralization.
- sphere_ray_intersection: find the ray-sphere intersection.
- surface_ray_intersection: find the ray-surface intersection. Support `sphere_tracing`/`root_finding`.
- aabb_ray_intersection: find the ray-volume intersection.

![rays](../assets/geometry/rays.gif)

------------------------------------------------------------------------
# triangle
Functions for triangle calculation.
- get_tri_normal
- get_tri_circumcircle

![tri](../assets/geometry/tri.gif)
![tri_norm](../assets/geometry/tri_norm.gif)

------------------------------------------------------------------------
# mesh
Function for mesh extraction, color extraction, etc.
- extract_mesh: simple marching cubes on cpu
- save_meshes: save mesh to `.ply` files
- get_verts_by_faces: rearrange verts from `(V, 3)` to `(F, 3, 3)` by faces.
- get_normals: get the vert/face normals
- get_face_centers: get the triangle centers
- simplify_mesh: simplify mesh
- render_mesh: it will call open3d/pytorch3d backend to render the mesh by cam positions.


![mesh](../assets/geometry/mesh.gif)

------------------------------------------------------------------------
# volume
Definition and function of a volume. For all the point you can get it in grid or fatten(by default)
- n_grid: num of volume/line seg on each side
- You can set the size of the volume by `side` or `xyz_len`
- corner: 8 corner pts
- grid_pts: all the grid points, in total `(n_grid+1)^3`
- voxel_size: each voxel size, useful for marching cubes.
- volume_pts: all the volume center points, in total `(n_grid)^3`
  - This can be sent to the network and get the volume density
- voxel_pts: the partial volume pts of each voxel that selected
- bound_lines: outside bounding lines, `12` lines with `(2, 3)` start-end pts.
- dense_lines: inner+outside bounding lines, `3*(n+1)^3` lines with `(2, 3)` start-end pts.
- bound_faces: outside bounding faces, `6 faces`, tensor in `(6, 4, 3)` shape
- dense_faces: inner+outside bounding faces, tensor in `((n_grid+1)n_grid^2*3, 4, 3)` shape
- convert_flatten_index_to_xyz_index/convert_xyz_index_to_flatten_index: index conversion
# occupancy
You can manually set up an occupancy record for visual. This can help to save computation like ray sphere intersection.
- set_up_voxel_bitfield: it creates a bool tensor of `(n_grid, n_grid, n_grid)` for recording this voxel's occupancy.
  - You can reset or update the occupied voxels as well. And take the occupied voxel's line/face for visualization.
  - get_occupied_voxel_idx/get_occupied_voxel_pts/get_occupied_grid_pts/get_occupied_lines/get_occupied_faces:
  You can get the occupied voxel for visualization or other computation.
  - get_occupied_bounding_corner/range: Get the smallest volume that bound all the occ voxels.
- set_up_voxel_opafield: the opacity field is the float opacity value of each voxel in `(n_grid, n_grid, n_grid)`.
  - You can update it, and use it to update the occupancy.
## ray/pts in volume
For ray in pts in volume, we provide a lot of function like
- check_pts_in_grid_boundary: check pts in voxel
- get_voxel_idx_from_xyz: get voxel idx from pts position
- get_grid_pts_idx_by_voxel_idx/get_grid_pts_by_voxel_idx: get grid pts index and position by voxel idx
- cal_weights_to_grid_pts / interpolate: interpolate pts by grid_pts using trilinear interpolation
- ray_volume_intersection: call the aabb intersection test and find the ray-volume intersection
  - ray_volume_intersect_in_occ_voxel: You can call to find the intersection in occupied voxels only
- get_ray_pass_through: get the voxel_ids that the ray pass through


![ray_pass](../assets/geometry/ray_pass.gif)
![volume_bound_sample](../assets/geometry/volume_bound_sample.gif)

------------------------------------------------------------------------

# point cloud
Function of point cloud with pts and color.
- save_point_cloud: export pc file as .ply file

![pc](../assets/geometry/pc.gif)

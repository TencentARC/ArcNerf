# plot3d
We provide a `draw_3d_components` that is helpful to plot camera/points/rays/lines/sphere
with customized color in all.

For all point, it should be in world space, or transformed to world space.
Then change to plt space. The visual space is (x-right, y-down, z-forward).
But we show the y values as -y inorder to make y goes downward

## Backend
We provide `matplotlib` and `plotly` for visualization. By default it use `plt`.

`Plotly` is good for interactive visualization
and is able to zoom-in/out, move. We can add `plotly=True` in `draw_3d_components`.

## Camera
- c2w pose in `(N_cam, 4, 4)`. Local camera models will be created and transformed into world space
- cam_colors in `(N_cam, 3)` or `(3,)` can be used to color each or all cams
- intrinsic in `(3, 3)` is used to adjust local cam model for actual ray direction.

## Points
- point in `(N_p, 3)` in world space
- point_size is single value used to set each point size
- point_color in `(N_p, 3)` or `(3,)` can be used to color each or all points

## Lines
- lines: a list of lines in `(N_pts_in_line, 3)` in world space, total `N_line`,
each line prints `N_pt_in_line-1` line seg
- line_colors in `(N_line, 3)` or `(3,)` can be used to color each or all lines

## Meshes
- meshes: a list of mesh of `(N_tri, 3, 3)` in world space, len is `N_m`
-  mesh_colors: color in `(N_m, 3)` or (3,), applied for each or all mesh

## Rays
- rays: a tuple `(rays_o, rays_d)`, each in `(N_r, 3)`, in world coord. `rays_d` is with actual len, if you want longer arrow, you need to extend `rays_d`
- ray_colors: color in `(N_r, 3)` or `(3,)`, applied for each or all rays
- ray_linewidth: width of ray line, by default is 1

## Sphere
- sphere_radius: draw a sphere with such `radius` if not None
- sphere_origin: the origin of sphere, by default is `(0, 0, 0)`

## Voxel
- TBD

## Other:
- title: title of the fig
- save_path: if `None`, will show the fig, otherwise save it
- axis range: update by the component with max values (xyz), show in a cube with same lengths.


# plot2d
In `common.visual`, we provide a `draw_2d_components` that is
helpful to plot points/lines in defined colors and legends. We use `matplotlib` as backend.

# TODO:
- Write a python notebook showing how to use
- open3d version for pc, mesh visual with rays if necessary

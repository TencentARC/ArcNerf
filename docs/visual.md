# plot3d
We provide a `draw_3d_components` that is helpful to plot camera/points/rays/lines/sphere
with customized color in all.

## sphere
Many function about sphere is at `arcnerf/geometry/sphere`
Any point on a unit sphere with (0,0,0) origin can be represented by (u, v),
where u in (0, 2pi), v in (0, pi)
- x = cos(u) * sin(v)
- y = cos(v)
- z = sin(u) * sin(v)

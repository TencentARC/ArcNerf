# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio

from .camera_model import create_camera_model, get_cam_whf
from arcnerf.geometry.sphere import get_sphere_surface
from common.visual.draw_cv2 import get_colors


def transform_plt_space(pts, xyz_axis=0):
    """Transform any point in world space to plt space
    This will exchange y/z axis. y-direction is down(But use -y)

    Args:
        pts: np.array in (n_pts, 3) or (3, n_pts)
        xyz_axis: 0 or 1, 0 means pts in (3, n_pts), else (n_pts, 3)

    Returns:
        pts_rot: np.array with same shape as pts, exchange y/z coord

    """
    assert xyz_axis in [0, 1], 'invalid xyz_axis'
    assert len(pts.shape) == 2 and (pts.shape[0] == 3 or pts.shape[1] == 3), 'Shape should be (n, 3) or (3, n)'

    if xyz_axis == 1:
        pts = np.transpose(pts, [1, 0])  # (3, pts)

    # rot mat
    rot_mat = np.identity(3, dtype=pts.dtype)
    rot_mat[1, 1] = 0
    rot_mat[1, 2] = 1
    rot_mat[2, 1] = -1
    rot_mat[2, 2] = 0

    pts_rot = rot_mat @ pts
    if xyz_axis == 1:
        pts_rot = np.transpose(pts_rot, [1, 0])  # (pts, 3)

    return pts_rot


def colorize_np(color_np):
    """float np color to rgb() which is used for plotly

    Args:
        color_np: np(n, 3) or np(3,) array

    Returns:
        str of list of str with 'rgb(255, 255, 255)' format

    """
    if len(color_np.shape) == 1:
        colors = 'rgb({}, {}, {})'.format(int(color_np[0] * 255), int(color_np[1] * 255), int(color_np[2] * 255))
    else:
        colors = []
        for i in range(color_np.shape[0]):
            colors.append(
                'rgb({}, {}, {})'.format(
                    int(color_np[i, 0] * 255), int(color_np[i, 1] * 255), int(color_np[i, 2] * 255)
                )
            )

    return colors


def draw_cameras(ax, c2w, cam_colors, intrinsic, min_values, max_values, plotly):
    """Draw cameras"""
    # set color, by default is red
    n_cam = c2w.shape[0]
    if cam_colors is None:
        cam_colors = get_colors('red', to_int=False, to_np=True)
    if cam_colors.shape == (3, ):
        cam_colors = np.repeat(cam_colors[None, :], n_cam, axis=0)
    assert cam_colors.shape == (n_cam, 3), 'Invalid cam colors shape...(N_cam, 3) or (3,)'

    # set vis params, adjust by camera loc
    max_cam_pose_norm = np.linalg.norm(c2w[:, :3, 3], axis=-1).max()
    cam_width, cam_height, f_scale = get_cam_whf(intrinsic, max_cam_pose_norm)

    # single camera_model in local coord. Each is a xxx
    camera_model = create_camera_model(cam_width, cam_height, f_scale)

    for idx in range(c2w.shape[0]):  # each camera
        mat = c2w[idx]
        for i in range(len(camera_model)):  # each polygon. (4, n_pts)
            X = np.zeros(camera_model[i].shape, dtype=mat.dtype)
            # to world coord
            for j in range(X.shape[1]):  # each point in polygon, (4, )
                X[:4, j] = mat @ camera_model[i][:4, j]
            X = transform_plt_space(X[:3, :])
            # draw in world coord. plot3D plots line betweem neighbour vertices
            if plotly:
                ax.append(
                    go.Scatter3d(
                        x=X[0, :],
                        y=X[1, :],
                        z=X[2, :],
                        mode='lines',
                        line={'color': colorize_np(cam_colors[idx])},
                        showlegend=False,
                        name='cam {}'.format(i)
                    )
                )
            else:
                ax.plot3D(X[0, :], X[1, :], X[2, :], color=cam_colors[idx])  # draw multi lines
            min_values = np.minimum(min_values, X.min(1))
            max_values = np.maximum(max_values, X.max(1))

    return min_values, max_values


def draw_points(ax, points, point_colors, point_size, min_values, max_values, plotly):
    """Draw points"""
    # set color, by default is green
    n_p = points.shape[0]
    if point_colors is None:
        point_colors = get_colors('green', to_int=False, to_np=True)
    if point_colors.shape == (3, ):
        point_colors = np.repeat(point_colors[None, :], n_p, axis=0)
    assert point_colors.shape == (n_p, 3), 'Invalid point colors shape...(N_p, 3) or (3,)'

    points_plt = transform_plt_space(points, xyz_axis=1)
    if plotly:
        ax.append(
            go.Scatter3d(
                x=points_plt[:, 0],
                y=points_plt[:, 1],
                z=points_plt[:, 2],
                mode='markers',
                marker={
                    'color': colorize_np(point_colors),
                    'size': point_size / 5.0
                },
                showlegend=False
            )
        )
    else:
        ax.scatter3D(points_plt[:, 0], points_plt[:, 1], points_plt[:, 2], color=point_colors, s=point_size)
    min_values = np.minimum(min_values, points_plt.min(0))
    max_values = np.maximum(max_values, points_plt.max(0))

    return min_values, max_values


def draw_rays(ax, rays, ray_colors, ray_linewidth, min_values, max_values, plotly):
    """Draw rays"""
    # set color, by default is blue
    n_r = rays[0].shape[0]
    if ray_colors is None:
        ray_colors = get_colors('blue', to_int=False, to_np=True)
    if ray_colors.shape == (3, ):
        ray_colors = np.repeat(ray_colors[None, :], n_r, axis=0)
    assert ray_colors.shape == (n_r, 3), 'Invalid ray colors shape...(N_r, 3) or (3,)'

    rays_o = rays[0]
    rays_d = rays[1]
    rays_e = rays_o + rays_d
    rays_o_plt = transform_plt_space(rays_o, xyz_axis=1)
    rays_d_plt = transform_plt_space(rays_d, xyz_axis=1)
    rays_e_plt = transform_plt_space(rays_e, xyz_axis=1)
    if plotly:
        # line in arrow
        for idx in range(rays_o_plt.shape[0]):
            ax.append(
                go.Scatter3d(
                    x=np.concatenate([rays_o_plt[idx, 0:1], rays_e_plt[idx, 0:1]], axis=-1),
                    y=np.concatenate([rays_o_plt[idx, 1:2], rays_e_plt[idx, 1:2]], axis=-1),
                    z=np.concatenate([rays_o_plt[idx, 2:3], rays_e_plt[idx, 2:3]], axis=-1),
                    mode='lines',
                    line={
                        'color': colorize_np(ray_colors[idx]),
                        'width': ray_linewidth * 2.0
                    },
                    showlegend=False
                )
            )
        # cone in array
        cone_ratio = 1.0 / 50.0
        cone_start = rays_o_plt * cone_ratio + (1 - cone_ratio) * rays_e_plt
        cone_len = cone_ratio * (rays_d_plt)
        for idx in range(rays_o_plt.shape[0]):
            rgb = colorize_np(ray_colors[idx])
            ax.append(
                go.Cone(
                    x=[cone_start[idx, 0]],
                    y=[cone_start[idx, 1]],
                    z=[cone_start[idx, 2]],
                    u=[cone_len[idx, 0]],
                    v=[cone_len[idx, 1]],
                    w=[cone_len[idx, 2]],
                    sizemode='scaled',
                    showscale=False,
                    sizeref=2,
                    colorscale=[[0, rgb], [1, rgb]],
                )
            )
    else:
        for idx in range(rays_o_plt.shape[0]):
            ax.quiver(
                rays_o_plt[idx, 0],
                rays_o_plt[idx, 1],
                rays_o_plt[idx, 2],
                rays_d_plt[idx, 0],
                rays_d_plt[idx, 1],
                rays_d_plt[idx, 2],
                color=ray_colors[idx],
                linewidths=ray_linewidth,
            )
    min_values = np.minimum(np.minimum(min_values, rays_o_plt.min(0)), rays_e_plt.min(0))
    max_values = np.maximum(np.maximum(max_values, rays_o_plt.max(0)), rays_e_plt.max(0))

    return min_values, max_values


def draw_sphere(ax, radius, origin, min_values, max_values, plotly, color=None, alpha=0.1):
    """Draw transparent sphere"""
    x, y, z = get_sphere_surface(radius, origin)
    # change coordinate
    xyz = np.concatenate([x[..., None], y[..., None], z[..., None]], axis=-1).reshape(-1, 3)
    xyz_plt = transform_plt_space(xyz, xyz_axis=1).reshape(x.shape[0], x.shape[1], -1)
    x_plt, y_plt, z_plt = xyz_plt[..., 0], xyz_plt[..., 1], xyz_plt[..., 2]
    if plotly:
        color = color if color is not None else 'dark'
        rgb = colorize_np(get_colors(color, to_int=False, to_np=True))
        colorscale = [[0, rgb], [1, rgb]]
        ax.append(
            go.Surface(
                x=x_plt,
                y=y_plt,
                z=z_plt,
                surfacecolor=np.zeros_like(x_plt),
                opacity=alpha,
                showscale=False,
                colorscale=colorscale,
                name='sphere'
            )
        )
    else:
        color = color if color is not None else 'grey'
        ax.plot_surface(x_plt, y_plt, z_plt, rstride=4, cstride=4, color=color, linewidth=0, alpha=alpha)
    min_values = np.minimum(min_values, np.array([x_plt.min(0).min(0), y_plt.min(0).min(0), z_plt.min(0).min(0)]))
    max_values = np.maximum(max_values, np.array([x_plt.max(0).max(0), y_plt.max(0).max(0), z_plt.max(0).max(0)]))

    return min_values, max_values


def draw_lines(ax, lines, line_colors, min_values, max_values, plotly):
    """Draw lines. Each line in list is a np.array with shape (N_pt_in_line, 3)"""
    # set color, by default is dark
    n_line = len(lines)
    if line_colors is None:
        line_colors = get_colors('dark', to_int=False, to_np=True)
    if line_colors.shape == (3, ):
        line_colors = np.repeat(line_colors[None, :], n_line, axis=0)
    assert line_colors.shape == (n_line, 3), 'Invalid line colors shape...(N_line, 3) or (3,)'

    for idx, line in enumerate(lines):
        line_plt = transform_plt_space(line, xyz_axis=1)  # (N_pt, 3)
        if plotly:
            ax.append(
                go.Scatter3d(
                    x=line_plt[:, 0],
                    y=line_plt[:, 1],
                    z=line_plt[:, 2],
                    mode='lines',
                    line={'color': colorize_np(line_colors[idx])},
                    showlegend=False,
                    name='line {}'.format(idx)
                )
            )
        else:
            ax.plot(line_plt[:, 0], line_plt[:, 1], line_plt[:, 2], color=line_colors[idx])
        min_values = np.minimum(min_values, line_plt.min(0))
        max_values = np.maximum(max_values, line_plt.max(0))

    return min_values, max_values


def draw_meshes(ax, meshes, mesh_colors, min_values, max_values, plotly):
    """Draw meshes. Each mesh in list is a np.array with shape (N_tri, 3, 3)"""
    # set color, by default is red
    n_m = len(meshes)
    if mesh_colors is None:
        mesh_colors = get_colors('silver', to_int=False, to_np=True)
    if mesh_colors.shape == (3, ):
        mesh_colors = np.repeat(mesh_colors[None, :], n_m, axis=0)
    assert mesh_colors.shape == (n_m, 3), 'Invalid mesh colors shape...(N_m, 3) or (3,)'

    for idx, mesh in enumerate(meshes):
        n_tri = mesh.shape[0]
        mesh_plt = transform_plt_space(mesh.reshape(-1, 3), xyz_axis=1).reshape(n_tri, 3, -1)  # (N_tri, 3, 3)
        if plotly:
            ax.append(
                go.Mesh3d(
                    x=mesh_plt.reshape(-1, 3)[:, 0],  # (N_tri * 3)
                    y=mesh_plt.reshape(-1, 3)[:, 1],  # (N_tri * 3)
                    z=mesh_plt.reshape(-1, 3)[:, 2],  # (N_tri * 3)
                    i=[0 + i * 3 for i in range(n_tri)],
                    j=[1 + i * 3 for i in range(n_tri)],
                    k=[2 + i * 3 for i in range(n_tri)],
                    color=colorize_np(mesh_colors[idx])
                )
            )
        else:
            ax.add_collection3d(
                Poly3DCollection([mesh_plt[i] for i in range(n_tri)], facecolors=mesh_colors[idx], linewidths=1)
            )
        min_values = np.minimum(min_values, mesh_plt.reshape(-1, 3).min(0))
        max_values = np.maximum(max_values, mesh_plt.reshape(-1, 3).max(0))

    return min_values, max_values


def draw_3d_components(
    c2w=None,
    cam_colors=None,
    intrinsic=None,
    points=None,
    point_size=20,
    point_colors=None,
    lines=None,
    line_colors=None,
    rays=None,
    ray_colors=None,
    ray_linewidth=2,
    meshes=None,
    mesh_colors=None,
    sphere_radius=None,
    sphere_origin=(0, 0, 0),
    title='',
    save_path=None,
    plotly=False,
    plotly_html=False
):
    """draw 3d component, including cameras, points, rays, etc
    For any pts in world space, you need to transform_plt_space to switch yz axis
    You can specified color for different cam/ray/point.
    Support plotly visual which allows zoom-in/out.
    TODO: Draw voxel grids

    Args:
        c2w: c2w pose stack in in shape(N_cam, 4, 4). None means not visual
        cam_colors: color in (N_cam, 3) or (3,), applied for each or all cam
        intrinsic: intrinsic in (3, 3), adjust local cam model if not None
        points: point in (N_p, 3) shape in world coord
        point_size: size of point, by default set up 20
        point_colors: color in (N_p, 3) or (3,), applied for each or all point
        lines: line in list of (N_pts_in_line, 3), len is N_line
        line_colors: color in (N_line, 3) or (3,), applied for each or all line
        rays: a tuple (rays_o, rays_d), each in (N_r, 3), in world coord
                rays_d is with actual len, if you want longer arrow, you need to extend rays_d
        ray_colors: color in (N_r, 3) or (3,), applied for each or all ray
        ray_linewidth: width of ray line, by default is 2
        meshes: list of mesh of (N_tri, 3, 3), len is N_m
        mesh_colors: color in (N_m, 3) or (3,), applied for each or all mesh
        sphere_radius: if not None, draw a sphere with such radius
        sphere_origin: the origin of sphere, by default is (0, 0, 0)
        title: a string of figure title
        save_path: path to save the fig. None will only show fig
        plotly: If True, use plotly instead of plt. Can be zoom-in/out. By default False.
        plotly_html: If True and save_path is not True, save to .html file instead of .png, good for interactive
    """
    fig = None
    if plotly:
        ax = []
    else:
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.set_aspect('auto')

    # axis range
    min_values = np.zeros((3, 1))
    min_values = np.inf
    max_values = np.zeros((3, 1))
    max_values = -np.inf

    # draw components
    if c2w is not None:
        min_values, max_values = draw_cameras(ax, c2w, cam_colors, intrinsic, min_values, max_values, plotly)

    if points is not None:
        min_values, max_values = draw_points(ax, points, point_colors, point_size, min_values, max_values, plotly)

    if rays is not None:
        min_values, max_values = draw_rays(ax, rays, ray_colors, ray_linewidth, min_values, max_values, plotly)

    if sphere_radius is not None:
        min_values, max_values = draw_sphere(ax, sphere_radius, sphere_origin, min_values, max_values, plotly)

    if lines is not None:
        min_values, max_values = draw_lines(ax, lines, line_colors, min_values, max_values, plotly)

    if meshes is not None:
        min_values, max_values = draw_meshes(ax, meshes, mesh_colors, min_values, max_values, plotly)

    # set axis limit
    axis_scale_factor = 0.25  # extent scale by such factor
    min_values -= (max_values - min_values) * axis_scale_factor
    max_values += (max_values - min_values) * axis_scale_factor

    X_min, Y_min, Z_min = min_values[0], min_values[1], min_values[2]
    X_max, Y_max, Z_max = max_values[0], max_values[1], max_values[2]
    max_range = np.array([X_max - X_min, Y_max - Y_min, Z_max - Z_min]).max() / 2.0

    mid_x = (X_max + X_min) * 0.5
    mid_y = (Y_max + Y_min) * 0.5
    mid_z = (Z_max + Z_min) * 0.5

    if plotly:  # plotly
        fig = {
            'data': ax,
            'layout': {
                'title': {
                    'text': title,
                    'x': 0.5
                },
                'scene': {
                    'xaxis': {
                        'range': [mid_x - max_range, mid_x + max_range],
                        'title': 'x'
                    },
                    'yaxis': {
                        'range': [mid_y - max_range, mid_y + max_range],
                        'title': 'z'
                    },
                    'zaxis': {
                        'range': [mid_z - max_range, mid_z + max_range],
                        'title': '- y'
                    },
                    'aspectmode': 'cube'
                },
                'showlegend': False,
                'coloraxis': {
                    'showscale': False
                }
            }
        }

        if save_path:
            if plotly_html:
                html_path = save_path.split('.')[:-1]
                html_path.append('.html')
                html_path = '.'.join(html_path)
                pio.write_html(fig, html_path)
            else:
                pio.write_image(fig, save_path)
        else:
            pio.show(fig)

    else:  # plt
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
        ax.set_xlabel('x')
        ax.set_ylabel('z')
        ax.set_zlabel('- y')
        ax.set_title(title)

        if save_path:
            fig.savefig(save_path)
        else:
            plt.show()

        plt.close()

# -*- coding: utf-8 -*-

import numpy as np
import pyfqmr
import skimage
import trimesh

from arcnerf.geometry.transformation import normalize
from arcnerf.render.render_pytorch3d import RenderPytorch3d


def extract_mesh(sigma, level, volume_size, volume_len, grad_dir='descent'):
    """Extracting mesh from sigma level sets. It has the same coord system with world coord.
    Currently this is only a cpu implementation. In the future may support cpu

    Args:
        sigma: (N, N, N) np array
        level: isosurface value. For nerf is around 50.0, for sdf-methods is 0.0.
        volume_size: tuple of 3, each small volume size
        volume_len: tuple of 3, each axis len
        grad_dir: if 'descent': object sigma is large than level (NeRF: Sigma is density)
                  if 'ascent': object sigma is smaller than level (volsdf: sdf is <0 inside)

    Returns:
        verts: (V, 3) np array, verts adjusted by volume offset
        faces: (F, 3) np array, faces
        vert_normals: (V, 3) np array, normal of each vert, pointing outside
    """
    assert grad_dir in ['descent', 'ascent'], 'Invalid grad_dir, only descent/ascent'
    verts, faces, vert_normals, _ = skimage.measure.marching_cubes(
        sigma, level=level, spacing=volume_size, gradient_direction=grad_dir
    )

    # adjust offset by whole large and small volume size
    volume_offset = ((volume_len[0] - volume_size[0]) / 2.0, (volume_len[1] - volume_size[1]) / 2.0,
                     (volume_len[2] - volume_size[2]) / 2.0)
    verts[:, 0] -= volume_offset[0]
    verts[:, 1] -= volume_offset[1]
    verts[:, 2] -= volume_offset[2]

    # normalize normals
    vert_normals = normalize(vert_normals)

    return verts, faces, vert_normals


def save_meshes(
    mesh_file, verts, faces, vert_colors=None, face_colors=None, vert_normals=None, face_normals=None, geo_only=False
):
    """Export mesh to .ply file

    Args:
        mesh_file: file path in .ply
        verts: (V, 3) np array, verts adjusted by volume offset
        faces: (F, 3) np array, faces
        vert_colors: (V, 3) np array, rgb color in (0~1). optional
        face_colors: (F, 3) np array, rgb color in (0~1). optional
        vert_normals: (V, 3) np array. normal of each vert, pointing outside, optional
        face_normals: (F, 3) np array. normal of each face, pointing outside, optional
        geo_only: If true, save verts/faces without color and normal. By default False
    """
    if geo_only:
        mesh_ply = trimesh.Trimesh(
            vertices=verts,
            faces=faces,
            vertex_normals=-vert_normals if vert_normals is not None else None,
            face_normals=-face_normals if face_normals is not None else None,
        )
        mesh_ply.export(mesh_file)
    else:
        vert_colors_uint8 = (255.0 * vert_colors.copy()).astype(np.uint8) if vert_colors is not None else None
        face_colors_uint8 = (255.0 * face_colors.copy()).astype(np.uint8) if face_colors is not None else None
        mesh_ply = trimesh.Trimesh(
            vertices=verts,
            faces=faces,
            vertex_normals=-vert_normals if vert_normals is not None else None,
            face_normals=-face_normals if face_normals is not None else None,
            vert_colors=vert_colors_uint8,
            face_colors=face_colors_uint8
        )
        mesh_ply.export(mesh_file)


def get_normals(verts, faces):
    """Get the face normals from verts/faces
    TOD: In fact, the normal of vert at edge and not very smooth. Many need to check if it affects anything

    Args:
        verts: (V, 3) np array, verts adjusted by volume offset
        faces: (F, 3) np array, faces

    Returns:
        vert_normals: (V, 3) np array, normal of each vert, pointing outside
        face_normals: (F, 3) np array, normal of each face, pointing outside
    """
    mesh = trimesh.Trimesh(vertices=verts, faces=faces)
    face_normals = -mesh.face_normals  # It get inside normals
    vert_normals = trimesh.geometry.mean_vertex_normals(verts.shape[0], faces, face_normals)

    # normalize
    vert_normals = normalize(vert_normals)
    face_normals = normalize(face_normals)

    return vert_normals, face_normals


def get_face_centers(verts, faces):
    """Get the face centers of each faces

    Args:
        verts: (V, 3) np array, verts adjusted by volume offset
        faces: (F, 3) np array, faces

    Returns:
        vert_normals: (V, 3) np array, normal of each vert
        face_centers: (F, 3) np array, center pos of each faces
    """
    mesh = trimesh.Trimesh(vertices=verts, faces=faces)

    return mesh.triangles_center


def get_verts_by_faces(verts, faces, vert_colors=None):
    """Rearrange verts in (V, 3) into (F, 3, 3) by getting the faces
    Verts_colors are map to faces, each use the mean color of three

    Args:
        verts: (V, 3) np array, verts adjusted by volume offset
        faces: (F, 3) np array, faces
        verts_colors: (V, 3) np array, rgb color in (0~1). optional

    Returns:
        verts_by_faces: (F, 3, 3) np array
        get_verts_by_faces: (F, 3), color of each face
    """
    verts_by_faces = np.take(verts, faces, axis=0)

    mean_face_colors = None
    if vert_colors is not None:
        mean_face_colors = np.take(vert_colors, faces, axis=0).mean(1)

    return verts_by_faces, mean_face_colors


def simplify_mesh(verts, faces, max_faces):
    """Simplify mesh by limit max_faces

    Args:
        verts: (V, 3) np array, verts adjusted by volume offset
        faces: (F, 3) np array, faces
        max_faces: max num of faces to keep

    Returns:
        verts: (V, 3) np array, verts adjusted by volume offset
        faces: (F, 3) np array, faces
    """
    n_f = faces.shape[0]
    if n_f < max_faces:
        return verts, faces

    # set up simplifier
    simplifier = pyfqmr.Simplify()
    simplifier.setMesh(verts, faces)
    simplifier.simplify_mesh(target_count=max_faces, aggressiveness=7, preserve_border=True, verbose=False)
    verts_sim, faces_sim, _ = simplifier.getMesh()

    return verts_sim, faces_sim


def render_mesh_images(
    verts,
    faces,
    vert_colors,
    face_colors,
    vert_normals,
    face_normals,
    H: int,
    W: int,
    w2c,
    intrinsic,
    backend,
    single_image_mode=False,
    device='cpu',
    sil_mode=False,
    sil_hard=True
):
    """Render mesh images

    Args:
        verts: (V, 3) np array, verts adjusted by volume offset
        faces: (F, 3) np array, faces
        vert_colors: (V, 3) np array, rgb color in (0~1). optional
        face_colors: (F, 3) np array, rgb color in (0~1). optional
        vert_normals: (V, 3) np array. normal of each vert, pointing outside, optional
        face_normals: (F, 3) np array. normal of each face, pointing outside, optional
        H: image height
        W: image width
        w2c: (N, 4, 4) np array, extrinsic
        intrinsic: (3, 3) np array, intrinsic
        backend: method to render image. Support 'open3d' and 'pytorch3d' now.
                 for open3d, it only support geo mode without colors
        single_image_mode: If True, render each image at once to save GPU memory.
                           Only in 'pytorch3d'. By default False.
        device: torch.device('cpu') or torch.device('cuda'). Only in 'pytorch3d'.
        sil_mode: whether to render in sil mode. Only in 'pytorch3d'. By default False.
        sil_hard: If in sil mode, render the silhouette in {0, 1} rather than (0-1).
                  By default True, Only in 'pytorch3d'.

    Returns:
        images: (N, H, W, 3) numpy array in uint8 from 0-255 in rgb order.
    """
    if backend == 'open3d':
        images = render_open3d(verts, faces, vert_normals, face_normals, H, W, w2c, intrinsic)
    elif backend == 'pytorch3d':
        if single_image_mode:
            renderer = RenderPytorch3d(
                H, W, batch_size=1, device=device, silhouette_mode=sil_mode, silhouette_hard=sil_hard
            )
            renderer.construct_by_matrix(intrinsic)
            images = []
            for i in range(w2c.shape[0]):
                images.append(
                    renderer.render(verts, faces, vert_colors, face_colors, vert_normals, face_normals, w2c[i:i + 1])
                )
            images = np.concatenate(images, axis=0)
        else:
            renderer = RenderPytorch3d(
                H, W, batch_size=w2c.shape[0], device=device, silhouette_mode=sil_mode, silhouette_hard=sil_hard
            )
            renderer.construct_by_matrix(intrinsic)
            images = renderer.render(verts, faces, vert_colors, face_colors, vert_normals, face_normals, w2c)
    else:
        raise NotImplementedError('Invalid render backend {}'.format(backend))

    return images


def render_open3d(verts, faces, vert_normals, face_normals, H: int, W: int, w2c, intrinsic):
    """Render image in open3d"""
    try:
        import open3d  # in case some machine do not support
    except ImportError:
        raise ImportError('Can not import open3d...')

    # init mesh
    mesh = open3d.geometry.TriangleMesh()
    mesh.vertices = open3d.utility.Vector3dVector(verts)
    mesh.triangles = open3d.utility.Vector3iVector(faces)
    mesh.vertex_normals = open3d.utility.Vector3dVector(vert_normals)
    mesh.triangle_normals = open3d.utility.Vector3dVector(face_normals)

    # set up scene
    vis = open3d.visualization.Visualizer()
    vis.create_window(width=W, height=H, visible=False)
    vis.add_geometry(mesh)

    # set intrinsic
    ctrl = vis.get_view_control()
    cam = ctrl.convert_to_pinhole_camera_parameters()
    intrinsic_adj = intrinsic.copy()
    intrinsic_adj[:2, 2] -= 0.5
    cam.intrinsic.intrinsic_matrix = intrinsic_adj
    ctrl.convert_from_pinhole_camera_parameters(cam)

    images = []
    for cam_id in range(w2c.shape[0]):
        cam.extrinsic = w2c[cam_id]
        ctrl.convert_from_pinhole_camera_parameters(cam)
        vis.poll_events()
        vis.update_renderer()
        image = vis.capture_screen_float_buffer(do_render=True)
        images.append((np.asarray(image) * 255.0).astype(np.uint8)[None])
    images = np.concatenate(images, axis=0)

    return images

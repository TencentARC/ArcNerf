# -*- coding: utf-8 -*-

import numpy as np
import torch

try:
    from pytorch3d.renderer import (
        MeshRasterizer, MeshRenderer, PerspectiveCameras, RasterizationSettings, SoftPhongShader, SoftSilhouetteShader,
        TexturesVertex, TexturesAtlas
    )
    from pytorch3d.structures import Meshes
except Exception:
    import warnings
    warnings.warn('Do not use pytorch3d render...')

from common.utils.torch_utils import torch_to_np
from common.visual import get_colors


class RenderPytorch3d:
    """
    A renderer using pytorch3d backend.
    Support silhouette mode and color mode
    TODO: This render is not only for visual, lightning/texture are not fully tested.
    TODO: May not be good for direct training
    """

    def __init__(
        self,
        h,
        w,
        dtype=torch.float32,
        batch_size=1,
        silhouette_mode=False,
        silhouette_hard=False,
        sigma=1e-5,
        device=torch.device('cpu'),
        to_np=True
    ):
        """
        Args:
            h: image height
            w: image width
            dtype: default torch dtype
            batch_size: num of image for rendering
            silhouette_mode: render in silhouette mode or not. By default False
            silhouette_hard: whether to hard cast silhouette image to {0, 1}, by default False
            sigma: use in sil raster_settings
            device: torch.device('cpu') or torch.device('cudax'). By default cpu.
            to_np: If True, turn final image into np image. False only in training mode
        """
        self.type = 'pytorch3d'
        self.resolution = (h, w)
        self.camera = None
        self.dtype = dtype
        self.batch_size = batch_size
        self.silhouette_mode = silhouette_mode
        self.silhouette_hard = silhouette_hard
        self.sigma = sigma
        self.device = device
        self.to_np = to_np

    def construct_by_matrix(self, intrinsic):
        """Construct intrinsic matrix by given a matrix

        Args:
            intrinsic: (B, 3, 3) or (3, 3) matrix, w2c matrix in torch or numpy
        """
        _intrinsic = self.handle_input(intrinsic, handle_device=False)
        focal = torch.cat([_intrinsic[:, 0, 0].unsqueeze(-1), _intrinsic[:, 1, 1].unsqueeze(-1)], dim=-1)  # (B, 2)
        center = torch.cat([_intrinsic[:, 0, 2].unsqueeze(-1), _intrinsic[:, 1, 2].unsqueeze(-1)], dim=-1)  # (B, 2)
        self.construct_by_focal_and_center(focal, center)

    def construct_by_focal_and_center(self, focal, center):
        """Construct intrinsic matrix by given a focal and center
        focal, center is in (x, y) / (w, h) order

        Args:
            focal: a tensor of size (B, 2)
            center: a tensor of size (B, 2)
        """
        _focal = self.handle_input(focal, expand_batch=False, handle_device=False)
        _center = self.handle_input(center, expand_batch=False, handle_device=False)

        image_size = torch.ones_like(_focal, dtype=_focal.dtype)
        image_size[:, 0] *= self.resolution[0]
        image_size[:, 1] *= self.resolution[1]
        self.camera = PerspectiveCameras(
            focal_length=_focal,  # -?
            principal_point=_center,
            image_size=image_size,
            in_ndc=False,
            device=self.device
        )

    def _setup_raster(self, img_shape):
        """Setup the raster for both mode"""
        if self.silhouette_mode:
            self.raster_settings = RasterizationSettings(
                image_size=img_shape,
                blur_radius=np.log(1. / 1e-4 - 1.) * self.sigma,
                faces_per_pixel=50,
            )
            self.shader = SoftSilhouetteShader()
        else:
            self.raster_settings = RasterizationSettings(
                image_size=img_shape,
                blur_radius=0.0,
                faces_per_pixel=1,
            )
            self.shader = SoftPhongShader(device=self.device, cameras=self.camera)

        self.renderer = MeshRenderer(
            rasterizer=MeshRasterizer(cameras=self.camera, raster_settings=self.raster_settings), shader=self.shader
        )

    def handle_input(self, value, expand_batch=True, handle_device=True):
        """Handle a single value"""
        if value is not None:
            _value = torch.tensor(value.copy(), dtype=self.dtype) if isinstance(value, np.ndarray) \
                else value.clone().type(self.dtype)

            if expand_batch and len(_value.shape) == 2:
                _value = torch.repeat_interleave(_value[None], self.batch_size, 0)

            if handle_device:
                _value = _value.to(self.device)
        else:
            _value = value

        return _value

    def handle_render_components(self, verts, faces, vert_colors, face_colors, vert_normals, face_normals, w2c):
        """get correct dtype and device
        All tensor has shape (B, n, 3) or (n, 3)
        """
        _verts = self.handle_input(verts)
        _faces = self.handle_input(faces)
        _vert_colors = self.handle_input(vert_colors)
        _face_colors = self.handle_input(face_colors)
        _vert_normals = self.handle_input(vert_normals)
        _face_normals = self.handle_input(face_normals)
        _w2c = self.handle_input(w2c)

        return _verts, _faces, _vert_colors, _face_colors, _vert_normals, _face_normals, _w2c

    def render(self, verts, faces, vert_colors=None, face_colors=None, vert_normals=None, face_normals=None, w2c=None):
        """The core rendering function. Allows back-propagation. Batch processing
        TODO: The rendered seems to have different w2c compared with rendering. Need to check.

        Args:
            verts: (B, n_vert, 3), tensor or np
            faces: (B, n_face, 3), tensor or np
            vert_colors: (B, n_vert, 3) or None, rgb value in float
            face_colors: (B, n_face, 3) or None, rgb value in float
            vert_normals: (B, n_vert, 3) or None, normal values
            face_normals: (B, n_face, 3) or None, normal values
            w2c: (B, 4, 4) or None, tensor or np. If None, use eyes

        Returns
            render_img: (B, H, W, 3) if rgb mode, (B, H, W) if sil mode. All in (0-255) uint8.
        """
        _verts, _faces, _vert_colors, _face_colors, _vert_normals, _face_normals, _w2c = self.handle_render_components(
            verts, faces, vert_colors, face_colors, vert_normals, face_normals, w2c
        )

        # reverse the normals, point inward seems correct TODO: Need to check source code for normal
        _vert_normals *= -1.0
        _face_normals *= -1.0

        # load mesh
        assert len(_verts.shape) == 3, 'verts should be in shape (b, v, 3)'
        assert len(_faces.shape) == 3, 'Faces should be in shape (b, v, 3)'

        mesh = Meshes(verts=_verts, faces=_faces, verts_normals=_vert_normals)
        if not self.silhouette_mode:
            if _face_colors is not None:
                mesh.textures = TexturesAtlas(_face_colors.unsqueeze(-2).unsqueeze(-2))  # (B, F, 1, 1, 3)
            elif _vert_colors is not None:
                mesh.textures = TexturesVertex(verts_features=_vert_colors)
            else:
                vert_colors_silver = get_colors('silver', to_int=False, to_np=True)  # (3,)
                vert_colors_silver = vert_colors_silver[None].repeat(_verts.shape[1],
                                                                     0)[None].repeat(_verts.shape[0], 0)
                _vert_colors = self.handle_input(vert_colors_silver)  # (B, V, 3)
                mesh.textures = TexturesVertex(verts_features=_vert_colors)

        if _w2c is not None:
            rotation = _w2c[:, :3, :3].to(self.device)
            translate = _w2c[:, :3, 3].to(self.device)
        else:
            rotation = torch.repeat_interleave(torch.eye(3, dtype=self.dtype).unsqueeze(0), self.batch_size, dim=0)\
                .to(self.device)
            translate = torch.zeros((self.batch_size, 3), dtype=self.dtype, device=self.device)

        # set camera
        self.camera.R = rotation
        self.camera.T = translate

        # setup raster setting
        self._setup_raster(self.resolution)  # needs to support rectangle image in future

        render_img = self.renderer(mesh, cameras=self.camera)  # (B, H, W, 4)

        # select mode
        if self.silhouette_mode:
            img = render_img[..., 3]
            if self.silhouette_hard:  # hard cast the sil image
                img[img >= 0.5] = 1.0
                img[img < 0.5] = 0.0
            img = img * 255.0
        else:
            img = render_img[..., :3] * 255.0

        # should flip image in both x,y dim to match volume render
        img = torch.flip(img, dims=[1, 2])  # (B, H, W, 3)

        if self.to_np:  # for visual mode only
            img = torch_to_np(img).astype(np.uint8)

        return img

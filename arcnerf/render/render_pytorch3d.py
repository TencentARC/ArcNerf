# -*- coding: utf-8 -*-

import numpy as np
import torch

from pytorch3d.renderer import (
    MeshRasterizer, MeshRenderer, PerspectiveCameras, RasterizationSettings, SoftPhongShader, SoftSilhouetteShader,
    TexturesVertex, TexturesAtlas
)
from pytorch3d.structures import Meshes

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
        device=torch.device('cpu')
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

    def construct_by_matrix(self, intrinsic):
        """Construct intrinsic matrix by given a matrix

        Args:
            intrinsic: (B, 3, 3) or (3, 3) matrix, w2c matrix in torch or numpy
        """
        intrinsic = self.handle_input(intrinsic, handle_device=False)
        focal = torch.cat([intrinsic[:, 0, 0].unsqueeze(-1), intrinsic[:, 1, 1].unsqueeze(-1)], dim=-1)  # (B, 2)
        center = torch.cat([intrinsic[:, 0, 2].unsqueeze(-1), intrinsic[:, 1, 2].unsqueeze(-1)], dim=-1)  # (B, 2)
        self.construct_by_focal_and_center(focal, center)

    def construct_by_focal_and_center(self, focal, center):
        """Construct intrinsic matrix by given a focal and center
        focal, center is in (x, y) / (w, h) order

        Args:
            focal: a tensor of size (B, 2)
            center: a tensor of size (B, 2)
        """
        focal = self.handle_input(focal, expand_batch=False, handle_device=False)
        center = self.handle_input(center, expand_batch=False, handle_device=False)

        image_size = torch.ones_like(focal, dtype=focal.dtype)
        image_size[:, 0] *= self.resolution[0]
        image_size[:, 1] *= self.resolution[1]
        self.camera = PerspectiveCameras(
            focal_length=focal,  # -?
            principal_point=center,
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
            value = torch.tensor(value, dtype=self.dtype) if isinstance(value, np.ndarray) \
                else value.type(self.dtype)

            if expand_batch and len(value.shape) == 2:
                value = torch.repeat_interleave(value[None], self.batch_size, 0)

            if handle_device:
                value = value.to(self.device)
        return value

    def handle_render_components(self, verts, faces, vert_colors, face_colors, vert_normals, face_normals, w2c):
        """get correct dtype and device
        All tensor has shape (B, n, 3) or (n, 3)
        """
        verts = self.handle_input(verts)
        faces = self.handle_input(faces)
        vert_colors = self.handle_input(vert_colors)
        face_colors = self.handle_input(face_colors)
        vert_normals = self.handle_input(vert_normals)
        face_normals = self.handle_input(face_normals)
        w2c = self.handle_input(w2c)

        return verts, faces, vert_colors, face_colors, vert_normals, face_normals, w2c

    def render(self, verts, faces, vert_colors=None, face_colors=None, vert_normals=None, face_normals=None, w2c=None):
        """The core rendering function. Allows back-propagation. Batch processing

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
        verts, faces, vert_colors, face_colors, vert_normals, face_normals, w2c = self.handle_render_components(
            verts, faces, vert_colors, face_colors, vert_normals, face_normals, w2c
        )

        # load mesh
        assert len(verts.shape) == 3, 'verts should be in shape (b, v, 3)'
        assert len(faces.shape) == 3, 'Faces should be in shape (b, v, 3)'

        mesh = Meshes(verts=verts, faces=faces, verts_normals=vert_normals)
        if not self.silhouette_mode:
            if face_colors is not None:
                mesh.textures = TexturesAtlas(face_colors.unsqueeze(-2).unsqueeze(-2))  # (B, F, 1, 1, 3)
            elif vert_colors is not None:
                mesh.textures = TexturesVertex(verts_features=vert_colors)
            else:
                vert_colors = get_colors('silver', to_int=False, to_np=True)  # (3,)
                vert_colors = vert_colors[None].repeat(verts.shape[1], 0)[None].repeat(verts.shape[0], 0)  # (B, V, 3)
                vert_colors = self.handle_input(vert_colors)
                mesh.textures = TexturesVertex(verts_features=vert_colors)

        if w2c is not None:
            rotation = w2c[:, :3, :3].clone().to(self.device)
            translate = w2c[:, :3, 3].clone().to(self.device)
        else:
            rotation = torch.repeat_interleave(torch.eye(3, dtype=self.dtype).unsqueeze(0), self.batch_size, dim=0)\
                .to(self.device)
            translate = torch.zeros((self.batch_size, 3), dtype=self.dtype).to(self.device)

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

        img = torch_to_np(img).astype(np.uint8)

        return img

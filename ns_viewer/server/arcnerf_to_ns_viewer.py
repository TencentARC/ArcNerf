# -*- coding: utf-8 -*-

import base64

import cv2
import torch
import torchvision

from ns_viewer.server.ns_utils import SceneBox


def arcnerf_dataset_to_ns_viewer(dataset):
    """Turn the arcnerf dataset into ns viewer"""
    ns_dataset = NSDataset(dataset)

    return ns_dataset


class NSDataset:
    """Nerf-studio dataset type"""

    def __init__(self, dataset):
        self.scene_box = SceneBox(aabb=torch.Tensor([[-1.5, -1.5, -1.5], [1.5, 1.5, 1.5]]))  # always default bbox now
        self.cameras = self.parse_cam(dataset)
        self.dataset = dataset
        self.pointcloud = self.parse_pc(dataset)

    def parse_cam(self, dataset):
        """Parse cam"""
        c2ws = []
        fx, fy, cx, cy = [], [], [], []
        n_img = len(dataset.cameras)
        for cam in dataset.cameras:
            c2w = cam.get_pose(torch_tensor=True)[None][:, :3, :4]  # (3, 4)
            c2w = self.arc_coord_to_ns_coord(c2w)
            c2ws.append(c2w)
            intrinsic = cam.get_intrinsic(torch_tensor=True)  # 3, 3
            fx.append(intrinsic[0, 0][None])  # (1)
            fy.append(intrinsic[1, 1][None])  # (1)
            cx.append(intrinsic[0, 2][None])  # (1)
            cy.append(intrinsic[1, 2][None])  # (1)

        c2ws = torch.cat(c2ws, dim=0)
        fx = torch.cat(fx, dim=0)[:, None]
        fy = torch.cat(fy, dim=0)[:, None]
        cx = torch.cat(cx, dim=0)[:, None]
        cy = torch.cat(cy, dim=0)[:, None]

        cam_dict = {
            '_field_custom_dimensions': {
                'camera_to_worlds': 2
            },
            'camera_to_worlds': c2ws,
            'fx': fx,
            'fy': fy,
            'cx': cx,
            'cy': cy,
            'distortion_params': None,
            'height': torch.ones((n_img, 1), dtype=torch.int64) * dataset.H,
            'width': torch.ones((n_img, 1), dtype=torch.int64) * dataset.W,
            'camera_type': torch.ones((n_img, 1), dtype=torch.int64),
            'times': None,
            '_shape': torch.Size([n_img])
        }

        return CamDict(cam_dict)

    def parse_pc(self, dataset):
        if 'pc' in dataset[0]:
            pc = dataset[0]['pc']
            pc_json = {
                'pts': arcnerf_pts_to_ns_viewer(pc['pts']).tolist(),  # (3n)
                'color': pc['color'].tolist(),  # (3n)
            }
            return pc_json
        else:
            return None

    def arc_coord_to_ns_coord(self, c2w):
        """Change coord of (1, 3, 4)"""
        return arcnerf_cam_to_ns_view(c2w[0])[None]

    def __getitem__(self, idx):
        out = {'image': torch.Tensor(self.dataset.images[idx])}
        return out

    def __len__(self):
        return len(self.dataset.cameras)


class CamDict:
    """Arcnerf cams to ns_viewer"""

    def __init__(self, cam_dict):
        self.cam_dict = cam_dict
        self.camera_to_world = self.cam_dict['camera_to_worlds']
        self.times = self.cam_dict['times']

    def to_json(self, camera_idx: int, image=None, max_size=None):
        """Write cams in arcnerf to json"""
        json_ = {
            'type': 'PinholeCamera',
            'cx': self.cam_dict['cx'][camera_idx, 0].item(),
            'cy': self.cam_dict['cy'][camera_idx, 0].item(),
            'fx': self.cam_dict['fx'][camera_idx, 0].item(),
            'fy': self.cam_dict['fy'][camera_idx, 0].item(),
            'camera_to_world': self.cam_dict['camera_to_worlds'][camera_idx].tolist(),
            'camera_index': camera_idx,
            'times': self.cam_dict['times'][camera_idx, 0].item() if self.times is not None else None,
        }

        if image is not None:
            image_uint8 = (image * 255).detach().type(torch.uint8)
            if max_size is not None:
                image_uint8 = image_uint8.permute(2, 0, 1)
                image_uint8 = torchvision.transforms.functional.resize(image_uint8, max_size)  # type: ignore
                image_uint8 = image_uint8.permute(1, 2, 0)
            image_uint8 = image_uint8.cpu().numpy()
            data = cv2.imencode('.jpg', image_uint8)[1].tobytes()
            json_['image'] = str('data:image/jpeg;base64,' + base64.b64encode(data).decode('ascii'))

        return json_


def ns_view_to_arcnerf_cam(c2w):
    """ns_viewer coord to arcnerf coord"""
    # make rotation correct
    c2w[2, 0] *= -1
    c2w[0:2, 1] *= -1
    c2w[0:2, 2] *= -1
    # make z downside up
    c2w[2, 3] *= -1
    # exchange y, x
    c2w = c2w[[0, 2, 1, 3], :]

    return c2w


def arcnerf_cam_to_ns_view(c2w):
    """Change coord in arcnerf as ns_viewer coord"""
    # exchange y, x
    c2w = c2w[[0, 2, 1], :]
    # make z downside up
    c2w[2, 3] *= -1
    # make rotation correct
    c2w[2, 0] *= -1
    c2w[0:2, 1] *= -1
    c2w[0:2, 2] *= -1

    return c2w


def arcnerf_pts_to_ns_viewer(pts):
    """Pts converts, (n, 3) shape"""
    pts = pts[:, [0, 2, 1]]
    pts[:, -1] *= -1

    return pts

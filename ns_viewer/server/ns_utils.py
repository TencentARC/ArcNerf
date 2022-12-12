# -*- coding: utf-8 -*-
# Copyright 2022 The Nerfstudio Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Func from nerfstudio"""

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from matplotlib import cm
import torch
from torchtyping import TensorType


def load_from_json(filename: Path):
    """Load a dictionary from a JSON filename.

    Args:
        filename: The filename to load from.
    """
    assert filename.suffix == '.json'
    with open(filename, encoding='UTF-8') as file:
        return json.load(file)


def write_to_json(filename: Path, content: dict):
    """Write data to a JSON file.

    Args:
        filename: The filename to write to.
        content: The dictionary data to write.
    """
    assert filename.suffix == '.json'
    with open(filename, 'w', encoding='UTF-8') as file:
        json.dump(content, file)


def get_dict_to_torch(stuff: Any, device: Union[torch.device, str] = 'cpu', exclude: Optional[List[str]] = None):
    """Set everything in the dict to the specified torch device.

    Args:
        stuff: things to convert to torch
        device: machine to put the "stuff" on
        exclude: list of keys to skip over transferring to device
    """
    if isinstance(stuff, dict):
        for k, v in stuff.items():
            if exclude and k in exclude:
                stuff[k] = v
            else:
                stuff[k] = get_dict_to_torch(v, device)
        return stuff
    if isinstance(stuff, torch.Tensor):
        return stuff.to(device)
    return stuff


# COLOR Func

WHITE = torch.tensor([1.0, 1.0, 1.0])
BLACK = torch.tensor([0.0, 0.0, 0.0])
RED = torch.tensor([1.0, 0.0, 0.0])
GREEN = torch.tensor([0.0, 1.0, 0.0])
BLUE = torch.tensor([0.0, 0.0, 1.0])

COLORS_DICT = {
    'white': WHITE,
    'black': BLACK,
    'red': RED,
    'green': GREEN,
    'blue': BLUE,
}


def get_color(color: Union[str, list]) -> TensorType[3]:
    """
    Args:
        color (Union[str, list]): Color as a string or a rgb list

    Returns:
        TensorType[3]: Parsed color
    """
    if isinstance(color, str):
        color = color.lower()
        if color not in COLORS_DICT:
            raise ValueError(f'{color} is not a valid preset color')
        return COLORS_DICT[color]
    if isinstance(color, list):
        if len(color) != 3:
            raise ValueError(f'Color should be 3 values (RGB) instead got {color}')
        return torch.tensor(color)

    raise ValueError(f'Color should be an RGB list or string, instead got {type(color)}')


def apply_colormap(image, cmap='viridis'):
    """Convert single channel to a color image.

    Args:
        image: Single channel image.
        cmap: Colormap for image.

    Returns:
        TensorType: Colored image
    """

    colormap = cm.get_cmap(cmap)
    colormap = torch.tensor(colormap.colors).to(image.device)  # type: ignore
    image_long = (image * 255).long()
    image_long_min = torch.min(image_long)
    image_long_max = torch.max(image_long)
    assert image_long_min >= 0, f'the min value is {image_long_min}'
    assert image_long_max <= 255, f'the max value is {image_long_max}'
    return colormap[image_long[..., 0]]


def apply_depth_colormap(
    depth,
    accumulation=None,
    near_plane: Optional[float] = None,
    far_plane: Optional[float] = None,
    cmap='turbo',
):
    """Converts a depth image to color for easier analysis.

    Args:
        depth: Depth image.
        accumulation: Ray accumulation used for masking vis.
        near_plane: Closest depth to consider. If None, use min image value.
        far_plane: Furthest depth to consider. If None, use max image value.
        cmap: Colormap to apply.

    Returns:
        Colored depth image
    """

    near_plane = near_plane or float(torch.min(depth))
    far_plane = far_plane or float(torch.max(depth))

    depth = (depth - near_plane) / (far_plane - near_plane + 1e-10)
    depth = torch.clip(depth, 0, 1)
    # depth = torch.nan_to_num(depth, nan=0.0) # TODO(ethan): remove this

    colored_image = apply_colormap(depth, cmap=cmap)

    if accumulation is not None:
        colored_image = colored_image * accumulation + (1 - accumulation)

    return colored_image


def apply_boolean_colormap(
    image,
    true_color=WHITE,
    false_color=BLACK,
):
    """Converts a depth image to color for easier analysis.

    Args:
        image: Boolean image.
        true_color: Color to use for True.
        false_color: Color to use for False.

    Returns:
        Colored boolean image
    """

    colored_image = torch.ones(image.shape[:-1] + (3, ))
    colored_image[image[..., 0], :] = true_color
    colored_image[~image[..., 0], :] = false_color
    return colored_image


@dataclass
class SceneBox:
    """Data to represent the scene box."""

    aabb: TensorType[2, 3] = None
    """aabb: axis-aligned bounding box.
    aabb[0] is the minimum (x,y,z) point.
    aabb[1] is the maximum (x,y,z) point."""

    def to_json(self) -> Dict:
        """Returns a json object from the Python object."""
        return {'type': 'aabb', 'min_point': self.aabb[0].tolist(), 'max_point': self.aabb[1].tolist()}

    @staticmethod
    def from_json(json_: Dict) -> 'SceneBox':
        """Returns the an instance of SceneBox from a json dictionary.

        Args:
            json_: the json dictionary containing scene box information
        """
        assert json_['type'] == 'aabb'
        aabb = torch.tensor([json_[0], json_[1]])
        return SceneBox(aabb=aabb)

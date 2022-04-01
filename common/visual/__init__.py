# -*- coding: utf-8 -*-

from matplotlib import cm
import numpy as np

colors = {
    'blue': [0, 0, 1],
    'navy': [0, 0, 0.5],
    'aqua': [0, 1, 1],
    'sky_blue': [0.529, 0.808, 0.922],
    'green': [0, 1, 0],
    'yellow_green': [0.604, 0.804, 0.196],
    'red': [1, 0, 0],
    'maroon': [0.5, 0, 0],
    'yellow': [1, 1, 0],
    'gold': [1, 0.843, 0],
    'orange': [1, 0.647, 0],
    'neutral': [.9, .9, .8],
    'grey': [1, 1, 1],
    'dark': [.3, .3, .3],
    'chocolate': [0.824, 0.412, 0.118],
    'black': [0, 0, 0],
    'magenta': [1, 0, 1],
    'purple': [0.5, 0, 0.5],
    'violet': [0.93, 0.51, 0.93],
    'olive': [0.5, 0.5, 0],
    'wheat': [0.961, 0.871, 0.701],
    'white': [1, 1, 1],
}


def get_colors(color='dark', to_int=True, opa=None, to_np=False):
    """Get the color in list with opa if required

    Args:
        color: color name
        to_int: If true, change range from (0, 1)-float64 to (0, 255)-int8
        opa: opacity value in (0, 1)
        to_np: if True, get numpy array

    Return:
        col: rgb value or rgba value in list, or np.ndarray if to_np
    """
    col = colors[color].copy()
    if to_int:
        col = [int(c * 255) for c in col]
    if opa is not None:
        col.append(opa)
    if to_np:
        col = np.array(col, dtype=np.int8 if to_int else np.float64)

    return col


def get_colors_from_cm(num, cm_func=cm.jet, opa=None, to_np=False):
    """Get colors from color_map

    Args:
        num: total num of different color,
        cm_func: color map function, by default is cm.jet
        opa: opacity value in (0, 1), applied to all
        to_np: if True, get numpy array

    Returns:
        a list of num of rgb color, or np.array in (num, 3)
    """
    cm_sub = np.linspace(0.0, 1.0, num)
    colors = [list(cm_func(x)) for x in cm_sub]  # (n, (4)) list
    if opa:
        for i in range(len(colors)):
            colors[i][-1] = opa
    else:
        colors = [col[:3] for col in colors]

    if to_np:
        colors = np.concatenate([colors])

    return colors


def get_combine_colors(colors, n_samples, to_int=True, opa=None):
    """Get combined colors from colors list and n_samples list

    Args:
        colors: list of color in str, for example ['red', 'blue']
        n_samples: list of num for each color [3, 5]
        to_int: If true, change range from (0, 1)-float64 to (0, 255)-int8
        opa: opacity value in (0, 1)

    Returns:
        colors: np(c1+c2+...cn, 3), all color combined. By default is np.array
    """
    color_list = []
    for c, n in zip(colors, n_samples):
        color = np.repeat(get_colors(c, to_int=False, to_np=True)[None, :], n, axis=0)
        color_list.append(color)
    colors = np.concatenate(color_list, axis=0)

    return colors

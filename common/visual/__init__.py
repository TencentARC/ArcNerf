# -*- coding: utf-8 -*-

from matplotlib import cm
import numpy as np

colors = {
    'blue': [0, 0, 1.0],
    'green': [0, 1.0, 0],
    'neutral': [.9, .9, .8],
    'grey': [1.0, 1.0, 1.0],
    'red': [1.0, 0, 0],
    'yellow': [.7, .75, .5],
    'dark': [.3, .3, .3],
}


def get_colors(color='dark', to_int=True, opa=None, to_np=False):
    """Get the color in list with opa if required

    Args:
        color: color name
        to_int: If true, change range from (0, 1) to (0, 255)
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
        col = np.array(col, dtype=np.int if to_int else np.float)

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

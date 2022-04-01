import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

import numpy as np

def get_rot(h):
    return np.array([
        [np.cos(h), np.sin(h)],
        [-np.sin(h), np.cos(h)],
    ])

def get_corners(box, lw):
    l, w = lw
    simple_box = np.array([
        [-l/2., -w/2.],
        [l/2., -w/2.],
        [l/2., w/2.],
        [-l/2., w/2.],
    ])
    h = np.arctan2(box[3], box[2])
    rot = get_rot(h)
    simple_box = np.dot(simple_box, rot)
    simple_box += box[:2]
    return simple_box


def plot_box(box, lw, color='g', alpha=0.7, no_heading=False):
    l, w = lw
    h = np.arctan2(box[3], box[2])
    simple_box = get_corners(box, lw)

    arrow = np.array([
        box[:2],
        box[:2] + l/2.*np.array([np.cos(h), np.sin(h)]),
    ])

    plt.fill(simple_box[:, 0], simple_box[:, 1], color=color, edgecolor='k',
             alpha=alpha, zorder=3, linewidth=1.0)
    if not no_heading:
        plt.plot(arrow[:, 0], arrow[:, 1], color=color, alpha=0.5)


def plot_car(x, y, h, l, w, color='b', alpha=0.5, no_heading=False):
    plot_box(np.array([x, y, np.cos(h), np.sin(h)]), [l, w],
             color=color, alpha=alpha, no_heading=no_heading)
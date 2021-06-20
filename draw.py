#!/usr/bin/env python3

from base import Map
import numpy as np

from matplotlib.patches import Circle
from matplotlib.collections import PatchCollection, LineCollection
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.lines import Line2D

from prm import PRM


def draw_ballsmap_2d(bm: 'BallsMap', ax: plt.Axes):
    patches = []
    for (x, y), sqr in zip(bm.position, bm.sq_radius):
        patches.append(Circle((x, y), np.sqrt(sqr)))
    col = PatchCollection(patches, color='k')
    ax.add_collection(col)


def draw_cspace_slice(x0: np.ndarray, min_bound: np.ndarray,
                      max_bound: np.ndarray, m: Map, ax: plt.Axes,
                      num_sample: int = 16):
    ndim = len(min_bound)

    for a in range(ndim):
        # slice along bounds
        lo = x0.copy()
        hi = x0.copy()
        lo[a] = min_bound[a]
        hi[a] = max_bound[a]

        x = np.linspace(lo, hi, num_sample)
        v = np.linspace(min_bound[a], max_bound[a], num_sample)
        occ = m.query(x)

        y = v[occ]
        n = v[~occ]
        # occs.append(v[occ])
        # free.append(v[~occ])
        ax.plot(np.full_like(n, a), n, 'rx')
        ax.plot(np.full_like(y, a), y, 'b+')
    ax.grid()


def animate_prm_edges(aux_edges, project,
                      fig: plt.Figure, out_file: str = '/tmp/prm_edge.mp4'):
    vert = aux_edges['v']
    nbr = aux_edges['nbr']
    con = aux_edges['con']
    v = project(vert)

    ax = fig.gca()
    ax.set_title('PRM edge construction')

    h, = ax.plot(v[..., 0], v[..., 1], 'm.', label='nodes')

    col_pos = LineCollection([], alpha=0.25, color='b')
    ax.add_collection(col_pos)

    col_neg = LineCollection([], alpha=0.25, color='r')
    ax.add_collection(col_neg)

    def make_proxy(color, **kwds):
        return Line2D([0, 1], [0, 1], color=color, **kwds)
    proxies = [make_proxy(color) for color in ['b', 'r']] + [h]
    ax.legend(proxies, ['pos', 'neg', 'node'])
    # plt.legend()

    def _init():
        return (col_pos, col_neg)

    def _update(i):
        c = con[:i]
        pos = v[nbr[:i][c]]
        neg = v[nbr[:i][~c]]
        col_pos.set_segments(pos)
        col_neg.set_segments(neg)
        return (col_pos, col_neg)

    anim = FuncAnimation(fig, _update, frames=np.arange(len(con)),
                         init_func=_init, blit=True)
    anim.save(out_file)
    plt.show()

#!/usr/bin/nev python3

from base import Map
import numpy as np
from typing import Tuple
from util import interp_axis

from matplotlib import pyplot as plt
from matplotlib import _pylab_helpers
from matplotlib.animation import FuncAnimation


class GridMap(Map):
    def __init__(self, data: np.ndarray, transform=None):
        if not isinstance(data, np.ndarray):
            data = np.asarray(data, dtype=np.bool)
        self.data = data
        self.transform = transform

    def query(self, pos: Tuple[float, ...]) -> bool:
        return self.data[pos]


class BallsMap(Map):
    def __init__(self, position: np.ndarray, radius: np.ndarray):
        self.position = position
        self.sq_radius = np.square(radius)

    def _map_to_xy(self, pos: np.ndarray) -> np.ndarray:
        return pos

    def query(self, pos: np.ndarray) -> np.ndarray:
        dim = self.position.shape[-1]
        pos = np.asarray(pos)
        out_shape = pos.shape[:-1]

        # batchwise processing
        pos = pos.reshape(-1, dim)
        delta = self.position[..., None, :] - pos
        sq_radius = np.einsum('...b,...b->...', delta, delta)  # N,M
        occupied = (sq_radius < self.sq_radius[:, None]).any(axis=0)
        return occupied.reshape(out_shape)

    @classmethod
    def generate(cls, bounds, min_obs_area, max_obs_area, num_obstacles):
        # Derived parameters
        area = np.prod(bounds[1] - bounds[0])
        max_obs_rad = np.sqrt(area * max_obs_area / np.pi)
        min_obs_rad = np.sqrt(area * min_obs_area / np.pi)

        # Insantiate `BallsMap`
        pos = np.random.uniform(
            bounds[0],
            bounds[1],
            size=(num_obstacles, bounds.shape[-1]))
        rad = np.abs(
            np.random.uniform(
                min_obs_rad,
                max_obs_rad,
                size=num_obstacles))
        return cls(pos, rad)


def get_old_fig():
    figManager = _pylab_helpers.Gcf.get_active()
    if figManager is not None:
        return figManager.canvas.figure
    else:
        return None


class PlottedMap(Map):
    def __init__(self, base_map: Map, ax: plt.Axes):
        self.base_map = base_map
        self.pos = None
        self.neg = None
        self.h_pos, = ax.plot([], [], 'b.', markersize=0.3)
        self.h_neg, = ax.plot([], [], 'r.', markersize=0.3)
        self.ax = ax
        self.pcounts = []
        self.ncounts = []

    def query(self, pos: np.ndarray) -> np.ndarray:
        occ = self.base_map.query(pos)
        old_fig = get_old_fig()

        # update plots
        if self.pos is None:
            self.pos = pos[~occ]
        else:
            self.pos = np.concatenate([self.pos, pos[~occ]])
        if self.neg is None:
            self.neg = pos[occ]
        else:
            self.neg = np.concatenate([self.neg, pos[occ]])
        self.pcounts.append(self.pos.shape[0])
        self.ncounts.append(self.neg.shape[0])
        self.h_pos.set_data(self.pos.T)
        self.h_neg.set_data(self.neg.T)
        self.ax.figure.canvas.draw()
        plt.pause(0.001)

        if old_fig is not None:
            # print('restore = {} -> {}'.format(self.ax.figure.number, old_fig.number))
            plt.figure(old_fig.number)

        return occ


class TrackedMap(Map):
    def __init__(self, base_map: Map):
        self.base_map = base_map
        self.pos = None
        self.neg = None
        self.pcounts = []
        self.ncounts = []
        self.track = True

    def set_track(self, track: bool):
        self.track = track

    def __getattr__(self, name):
        return getattr(self.base_map, name)

    def query(self, pos: np.ndarray) -> np.ndarray:
        occ = self.base_map.query(pos)

        if self.track:
            # update plots
            if self.pos is None:
                self.pos = pos[~occ]
            else:
                self.pos = np.concatenate([self.pos, pos[~occ]])
            if self.neg is None:
                self.neg = pos[occ]
            else:
                self.neg = np.concatenate([self.neg, pos[occ]])
            self.pcounts.append(self.pos.shape[0])
            self.ncounts.append(self.neg.shape[0])

        return occ

    def animate(self, fig: plt.Figure = None):
        if fig is None:
            fig = plt.gcf()
        ax = fig.gca()
        ax.set_title('PRM vertex construction')

        h_pos, = plt.plot([], [], 'b.', label='pos')
        h_neg, = plt.plot([], [], 'r.', label='neg')
        plt.legend()

        num_frames = 128
        pcv2 = np.floor(np.mgrid[0:len(self.pos)+1:num_frames*1j]).astype(np.int32)
        ncv2 = np.floor(np.mgrid[0:len(self.neg)+1:num_frames*1j]).astype(np.int32)
        # num_frames = len(self.pcounts)

        def _init():
            if False:
                vmin = np.minimum(self.pos.min(axis=0), self.neg.min(axis=0))
                vmax = np.maximum(self.pos.max(axis=0), self.neg.max(axis=0))
                ax.set_xlim(vmin[0], vmax[0])
                ax.set_ylim(vmin[1], vmax[1])
            return (h_pos, h_neg)

        def _update(frame):
            # i_p = self.pcounts[frame]
            # i_n = self.ncounts[frame]
            i_p = pcv2[frame]
            i_n = ncv2[frame]
            h_pos.set_data(self.pos[:i_p].T)
            h_neg.set_data(self.neg[:i_n].T)
            return (h_pos, h_neg)

        anim = FuncAnimation(fig, _update, frames=np.arange(num_frames),
                             init_func=_init, blit=True)
        anim.save('/tmp/prm_vert.mp4')
        plt.show()


class PlanarArmsMap(Map):
    """Models N-armed robot connected by infinitely thin links."""

    def __init__(self,
                 base_map: Map,
                 link_length: np.ndarray,
                 base_pose=(0, 0, 0),
                 num_interp: int = 16):
        self.base_map = base_map
        self.base_pose = base_pose  # (x,y,h)
        self.link_length = link_length
        self.num_interp = num_interp

    def _map_to_xy(self, pos: np.ndarray) -> np.ndarray:
        angles = self.base_pose[2] + np.cumsum(pos, axis=-1)  # ?, num_links
        offsets = self.link_length[..., None] * np.stack(
            [np.cos(angles), np.sin(angles)], axis=-1)
        positions = self.base_pose[:2] + np.cumsum(offsets, axis=-2)
        return positions

    def query(self, pos: np.ndarray) -> np.ndarray:
        pos = self._map_to_xy(pos)
        pos = np.insert(pos, 0, self.base_pose[:2], axis=-2)
        pos = interp_axis(pos, axis=-2, num=self.num_interp)
        out = self.base_map.query(pos).any(axis=-1)
        return out

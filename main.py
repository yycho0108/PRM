#!/usr/bin/env python3

from abc import ABC, abstractmethod
import numpy as np
from typing import Tuple, List, Dict, Hashable
from simple_parsing import Serializable
from dataclasses import dataclass
from scipy.spatial import cKDTree
import networkx as nx

from matplotlib import pyplot as plt
from matplotlib.collections import PatchCollection, LineCollection
from matplotlib.patches import Circle
from tqdm.auto import tqdm


def anorm(x):
    return (x + np.pi) % (2 * np.pi) - np.pi


def adiff(a, b):
    return anorm(a - b)


def in_bound(x, min_x, max_x, axis: int = -1):
    return np.logical_and(
        np.all(x >= min_x, axis=axis),
        np.all(x < max_x, axis=axis)
    )


def interp_axis(x, axis: int, residual_fn=None, *args, **kwargs):
    """interpolate an axis."""
    i_max = x.shape[axis] - 1
    w = np.linspace(0, i_max, *args, **kwargs)
    i0 = np.floor(w).astype(np.int32)
    i1 = np.minimum(i0 + 1, i_max)
    alpha = w - i0

    x0 = np.take(x, i0, axis=axis)
    x1 = np.take(x, i1, axis=axis)
    if residual_fn:
        dx = residual_fn(x0, x1)
    else:
        dx = x1 - x0
    out = x0 + alpha[:, None] * dx
    return out


@dataclass
class Settings(Serializable):
    pass


class Map(ABC):
    """Abstract base class for collision queries."""
    @abstractmethod
    def query(self, pos: Tuple[float, ...]) -> bool:
        return NotImplemented


class Sampler(ABC):
    """Abstract base class for sampling."""
    @abstractmethod
    def sample(self, shape: Tuple[int, ...]):
        return NotImplemented


class Planner(ABC):
    @abstractmethod
    def plan(self, q0, q1):
        return NotImplemented


class LinearLocalPlanner(Planner):
    def __init__(self, M: Map, N: int = 8):
        self.M = M
        self.N = N

    def plan(self, q0, q1):
        q = np.linspace(q0, q1, self.N, False)
        return ~self.M.query(q).any(axis=0)


class LinearAngleLocalPlanner(Planner):
    def __init__(self, M: Map, N: int = 8):
        self.M = M
        self.N = N

    def plan(self, q0, q1):
        dq = (q1 - q0 + np.pi) % (2 * np.pi) - np.pi
        q = q0 + np.linspace(0, dq, self.N, False)
        return ~self.M.query(q).any(axis=0)


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
        # TODO(ycho): pos should interpolate between arm joints
        out = self.base_map.query(pos).any(axis=-1)
        return out


class UniformGridSampler(Sampler):
    def __init__(self, min_bound, max_bound):
        self.min_bound = min_bound
        self.max_bound = max_bound

    def sample(self, shape: Tuple[int, ...]):
        if isinstance(shape, int):
            shape = (shape,)
        shape = tuple(shape) + (self.min_bound.shape[-1],)
        return np.random.uniform(
            low=self.min_bound,
            high=self.max_bound,
            size=shape)


class PRM:
    def __init__(self, S: Sampler, M: Map, P: Planner,
                 N: int = 128, K: int = 8):
        self.S = S
        self.M = M
        self.P = P

        self.N = N
        self.K = K

        # Graph
        self.V = None
        self.E = None
        self.G = None

        # Helpers
        self.D = None  # edge distances
        self.tree = None

    def _construct_vertices(self):
        N = self.N

        with tqdm(total=N) as pbar:
            count = 0
            while count < N:
                n = N - count
                q = self.S.sample(n)  # n,?
                occ = self.M.query(q)  # n
                q = q[~occ]

                # NOTE(ycho): lazy construction of V
                # based on runtime dimensions
                if self.V is None:
                    self.V = np.empty((N, q.shape[-1]), q.dtype)

                # Update V with newly queried stuff
                new_count = count + len(q)
                self.V[count: new_count] = q
                count = new_count
                pbar.update(len(q))

    def _construct_edges(self):
        self.tree = cKDTree(self.V)
        dist, inds = self.tree.query(self.V, k=self.K + 1)

        # Check connectivity between nodes
        src = np.arange(len(self.V))  # (N, 1)
        dst = inds[..., 1:]  # (N, K)
        con = self.P.plan(self.V[:, None, :], self.V[dst, :])
        print('con', con.shape)

        # if connected, then create edges between.
        src = np.broadcast_to(src[:, None], dst.shape)
        self.E = np.stack([src, dst], axis=-1)[con]
        self.D = dist[..., 1:][con]

    def _construct_graph(self):
        """Construct a networkx graph."""
        G = nx.Graph()
        G.add_weighted_edges_from(
            [(i0, i1, d) for(i0, i1), d in zip(self.E, self.D)],
            axis=-1)
        self.G = G

    def construct(self):
        self._construct_vertices()
        self._construct_edges()
        self._construct_graph()

    def query(self, q0, q1):
        q0 = np.asarray(q0)
        q1 = np.asarray(q1)
        _, (i0, i1) = self.tree.query([q0, q1], k=self.K)

        # Connect q0,q1 to roadmap
        src = np.stack([q0, q1])[:, None, :]  # 2,1,2
        dst = self.V[np.stack([i0, i1], axis=0), :]  # 2,4,2 == 2,K,D
        con = self.P.plan(src, dst)

        # Require that at least the connection was established.
        if not con.any(axis=-1).all():
            return None

        nbr = con.argmax(axis=-1)
        i0, i1 = i0[nbr[0]], i1[nbr[1]]
        # now to connect q0 -- i0 -- i1 -- q1
        try:
            p = nx.shortest_path(self.G, source=i0, target=i1)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return None
        return np.concatenate([q0[None], self.V[p], q1[None]], axis=0)


def draw_ballsmap_2d(bm: BallsMap, ax: plt.Axes):
    patches = []
    for (x, y), sqr in zip(bm.position, bm.sq_radius):
        # print(x, y, np.sqrt(sqr))
        patches.append(Circle((x, y), np.sqrt(sqr)))
    col = PatchCollection(patches, color='k')
    ax.add_collection(col)


def main():
    # seed = 9 # --> triggers nodenotfound
    # seed = 15  # semi-interesting path
    # seed = None
    # seed = 18
    # seed = np.random.randint(low=0, high=65536)
    seed = 8019
    print('seed', seed)
    np.random.seed(seed)
    bounds = np.asarray([[-10.0, -10.0], [10.0, 10.0]])  # (lo, hi)
    num_obstacles = 32
    max_obs_area = 0.03  # expressed in ratio w.r.t total map area
    min_obs_area = 0.01
    num_waypoints = 32
    num_vertices = 512
    num_neighbors = 8
    # robot_radius # << dilate obstacle by this much
    min_link_length = 0.3 * np.linalg.norm(bounds[1] - bounds[0], axis=-1)
    max_link_length = 0.05 * np.linalg.norm(bounds[1] - bounds[0], axis=-1)
    num_links = 4

    np.random.seed(seed)

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
    link_length = np.random.uniform(
        min_link_length,
        max_link_length,
        size=num_links)
    balls_map = BallsMap(pos, rad)

    if True:
        query_map = PlanarArmsMap(balls_map, link_length)
        sampler = UniformGridSampler(np.full(len(link_length), -np.pi),
                                     np.full(len(link_length), np.pi))
        local_planner = LinearAngleLocalPlanner(query_map, num_waypoints)
    else:
        query_map = balls_map
        sampler = UniformGridSampler(bounds[0], bounds[1])
        local_planner = LinearLocalPlanner(query_map, num_waypoints)

    prm = PRM(sampler, query_map, local_planner, num_vertices, num_neighbors)
    prm.construct()

    # Generate begin-end configurations.
    while True:
        q0 = None
        while True:
            q0 = sampler.sample(())
            if query_map.query(q0):
                continue
            q0_xy = query_map._map_to_xy(q0)
            if not in_bound(q0_xy, bounds[0], bounds[1]).all():
                continue
            break
        q1 = None
        while True:
            q1 = sampler.sample(())
            if query_map.query(q1):
                continue
            q1_xy = query_map._map_to_xy(q1)
            if not in_bound(q1_xy, bounds[0], bounds[1]).all():
                continue
            break
        path = prm.query(q0, q1)

        if path is not None:
            break

    # NOTE(ycho): map visualization
    if True:
        def _get_endpoint(q):
            if isinstance(query_map, PlanarArmsMap):
                return query_map._map_to_xy(q)[..., -1, :]
            else:
                return query_map._map_to_xy(q)

        ax = plt.gca()
        draw_ballsmap_2d(balls_map, ax)
        ax.set_xlim(2.0 * bounds[0, 0], 2.0 * bounds[1, 0])
        ax.set_ylim(2.0 * bounds[0, 1], 2.0 * bounds[1, 1])

        # V = prm.V
        V = _get_endpoint(prm.V)

        # show prm
        ax.plot(V[..., 0], V[..., 1], 'x', linewidth=4, markersize=8)

        # show PRM roadmap
        col = LineCollection(V[prm.E, :], alpha=0.25)
        ax.add_collection(col)

        # show init-goal pairs
        q0 = _get_endpoint(q0)
        q1 = _get_endpoint(q1)
        ax.plot(q0[0], q0[1], 'go')
        ax.plot(q1[0], q1[1], 'bo')

        # show path (if available)
        if path is not None:
            if isinstance(query_map, PlanarArmsMap):
                path_xy = query_map._map_to_xy(path)
                path_end = path_xy[:, -1]
                ax.plot(path_end[:, 0], path_end[:, 1], 'rx')

                path_up = interp_axis(path, axis=-2,
                                      residual_fn=lambda a, b: adiff(b, a)
                                      )
                path_xy = query_map._map_to_xy(path_up)
                path_end = path_xy[:, -1]
                ax.plot(path_end[:, 0], path_end[:, 1], 'r--')
            else:
                ax.plot(path[:, 0], path[:, 1], 'r--')

        ax.set_axisbelow(True)
        ax.grid(True)
        if isinstance(query_map, PlanarArmsMap):
            if path is not None:
                path_up = interp_axis(path, axis=-2,
                                      residual_fn=lambda a, b: adiff(b, a)
                                      )
                path_xy = query_map._map_to_xy(path_up)
                path_xy = np.insert(path_xy, 0, 0, axis=1)
                path_i = 0
                h, = ax.plot(
                    path_xy[path_i, ..., 0],
                    path_xy[path_i, ..., 1],
                    'm.-')
                while True:
                    path_i = (path_i + 1) % path_xy.shape[0]
                    h.set_data(
                        path_xy[path_i, ..., 0],
                        path_xy[path_i, ..., 1])
                    plt.pause(0.001)
        else:
            path_up = interp_axis(path, axis=-2)
            path_xy = query_map._map_to_xy(path_up)
            path_i = 0
            h, = ax.plot(
                path_xy[path_i, ..., 0],
                path_xy[path_i, ..., 1],
                'mo')
            while True:
                path_i = (path_i + 1) % path_xy.shape[0]
                h.set_data(
                    path_xy[path_i, ..., 0],
                    path_xy[path_i, ..., 1])
                plt.pause(0.001)


if __name__ == '__main__':
    main()

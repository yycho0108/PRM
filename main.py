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

    def _construct_edges(self):
        self.tree = cKDTree(self.V)
        dist, inds = self.tree.query(self.V, k=self.K + 1)

        # Check connectivity between nodes
        src = np.arange(len(self.V))  # (N, 1)
        dst = inds[..., 1:]  # (N, K)
        con = self.P.plan(self.V[:, None, :], self.V[dst, :])

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
    # m = GridMap([5, 5])
    # seed = 9 # --> triggers nodenotfound
    seed = 14  # semi-interesting path
    bounds = np.asarray([[-10.0, -10.0], [10.0, 10.0]])  # (lo, hi)
    num_obstacles = 32
    max_obs_area = 0.05  # expressed in ratio w.r.t total map area
    min_obs_area = 0.01
    num_waypoints = 8
    num_vertices = 128
    num_neighbors = 8
    # robot_radius # << dilate obstacle by this much

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
    query_map = BallsMap(pos, rad)
    sampler = UniformGridSampler(bounds[0], bounds[1])
    local_planner = LinearLocalPlanner(query_map, num_waypoints)

    prm = PRM(sampler, query_map, local_planner, num_vertices, num_neighbors)
    prm.construct()

    # Generate begin-end configurations
    q0 = None
    while True:
        q0 = sampler.sample(())
        if not query_map.query(q0):
            break
    q1 = None
    while True:
        q1 = sampler.sample(())
        if not query_map.query(q1):
            break
    path = prm.query(q0, q1)

    # NOTE(ycho): map visualization
    if True:
        ax = plt.gca()
        draw_ballsmap_2d(query_map, ax)
        ax.set_xlim(bounds[0, 0], bounds[1, 0])
        ax.set_ylim(bounds[0, 1], bounds[1, 1])

        # show prm
        ax.plot(prm.V[..., 0], prm.V[..., 1], '+')
        src = prm.V[prm.E[..., 0]]
        dst = prm.V[prm.E[..., 1]]

        # show PRM roadmap
        col = LineCollection(prm.V[prm.E, :])
        ax.add_collection(col)

        # show init-goal pairs
        ax.plot(q0[0], q0[1], 'go')
        ax.plot(q1[0], q1[1], 'bo')

        # show path (if available)
        if path is not None:
            ax.plot(path[:, 0], path[:, 1], 'r--')

        ax.set_axisbelow(True)
        ax.grid(True)
        plt.show()


if __name__ == '__main__':
    main()

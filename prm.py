#!/usr/bin/env python3

from typing import Tuple
import numpy as np
from base import Sampler, Map, Planner
from tqdm.auto import tqdm
from scipy.spatial import cKDTree
import networkx as nx


class PRM:
    def __init__(self, S: Sampler, M: Map, P: Planner,
                 N: int = 128, K: int = 8):
        self.sampler = S
        self.occupancy = M
        self.local_planner = P

        self.N = N  # num vertices
        self.K = K  # num neighbors

        # Graph
        self.V = None  # vertices
        self.E = None  # edges
        self.G = None  # graph

        # Helpers
        self.D = None  # edge distances
        self.tree = None  # KDTree for fast spatial neighborhood queries

    def _construct_vertices(self):
        N = self.N

        with tqdm(total=N) as pbar:
            count = 0
            while count < N:
                n = N - count
                q = self.sampler.sample(n)  # n,?
                occ = self.occupancy.query(q)  # n
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

    def _construct_edges(self, aux={}):
        self.tree = cKDTree(self.V)
        dist, inds = self.tree.query(self.V, k=self.K + 1)

        # Check connectivity between nodes
        src = np.arange(len(self.V))  # (N, 1)
        dst = inds[..., 1:]  # (N, K)
        con = self.local_planner.plan(self.V[:, None, :], self.V[dst, :])

        # if connected, then create edges between.
        src = np.broadcast_to(src[:, None], dst.shape)
        nbr = np.stack([src, dst], axis=-1)
        aux['v'] = self.V
        aux['nbr'] = nbr
        aux['con'] = con

        self.E = nbr[con]
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
        con = self.local_planner.plan(src, dst)

        # Require that at least one connection was established
        # on both ends.
        if not con.any(axis=-1).all():
            return None

        # Connect to roadmap, nearest preferred
        nbr = con.argmax(axis=-1)
        i0, i1 = i0[nbr[0]], i1[nbr[1]]

        # now to connect q0 -- i0 -- i1 -- q1
        try:
            p = nx.shortest_path(self.G, source=i0, target=i1)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return None
        return np.concatenate([q0[None], self.V[p], q1[None]], axis=0)

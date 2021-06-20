
import numpy as np
from base import Planner, Map


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

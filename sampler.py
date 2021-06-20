#!/usr/bin/env python3

from typing import Tuple
from base import Sampler
import numpy as np


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

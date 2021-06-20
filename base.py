#!/usr/bin/env python3

from typing import Tuple
from abc import ABC, abstractmethod


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

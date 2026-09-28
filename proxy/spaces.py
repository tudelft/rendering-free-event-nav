"""Minimal stand-ins for the two RL-framework types the environment declares.

The environment itself is framework-agnostic; it only needs a way to describe its
observation and action spaces and a base class to inherit from. Vendoring these here
keeps the repository runnable without pulling in a full agent framework.
"""
import numpy as np


class Space:
    """Dtype + shape descriptor, optionally bounded."""

    def __init__(self, dtype, shape=(), low=None, high=None):
        self.dtype = np.dtype(dtype)
        self.shape = tuple(shape) if isinstance(shape, (tuple, list)) else (shape,)
        if self.dtype == bool:
            low, high = False, True
        elif low is None and high is None and np.issubdtype(self.dtype, np.integer):
            low, high = np.iinfo(self.dtype).min, np.iinfo(self.dtype).max
        self.low = np.full(self.shape, low if low is not None else -np.inf, self.dtype) \
            if self.shape else low
        self.high = np.full(self.shape, high if high is not None else np.inf, self.dtype) \
            if self.shape else high

    @property
    def discrete(self):
        return np.issubdtype(self.dtype, np.integer) or self.dtype == bool

    def __repr__(self):
        return f'Space({self.dtype.name}, {self.shape})'


class Env:
    """Base class. Subclasses provide obs_space / act_space and step()."""

    def close(self):
        pass

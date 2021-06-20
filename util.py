#!/usr/bin/env python3

import numpy as np


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
    """interpolate along axis."""
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

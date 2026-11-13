"""Scalar profiles along the material coordinate of the fiber."""

from dataclasses import dataclass, field
from numbers import Real

import numpy as np

from ..backends import real_arrays


@dataclass(frozen=True, eq=False)
class SpatialProfile:
    """Piecewise linear profile, zero outside its finite knots [m].

    Positions are fixed Python or NumPy values, in nondecreasing order.
    A knot may occur twice to describe a right-continuous step.
    Values have the unit of the field (strain, K, index change, ...).
    They must be finite; only real dtype and shape are checked here,
    preserving native parameters for differentiation.
    """

    positions: object = field(repr=False)
    values: object = field(repr=False)

    def __post_init__(self):
        positions = tuple(self.positions)
        if any(isinstance(value, bool) or not isinstance(value, Real) for value in positions):
            raise TypeError("positions must contain fixed real Python or NumPy values")
        positions = np.array(positions, dtype=np.float64)
        if positions.size < 2 or not np.all(np.isfinite(positions)):
            raise ValueError("positions must contain at least two finite knots")
        widths = np.diff(positions)
        if np.any(widths < 0):
            raise ValueError("positions must be nondecreasing")
        if np.any((widths[:-1] == 0) & (widths[1:] == 0)):
            raise ValueError("a position may be repeated at most twice")
        xp, (values,) = real_arrays(self.values)
        if values.shape != positions.shape:
            raise ValueError("values must have one entry per position")
        positions.setflags(write=False)
        object.__setattr__(self, "positions", positions)
        object.__setattr__(self, "values", xp.asarray(values, copy=True))

    def value(self, position, *, xp=None):
        """Value at finite material positions [m], with any input shape."""
        xp, (position, values) = real_arrays(position, self.values, xp=xp)
        result = xp.zeros_like(position)
        for index in range(len(self.positions) - 1):
            left, right = float(self.positions[index]), float(self.positions[index + 1])
            if left == right:
                continue
            fraction = xp.clip(position - left, 0., right - left) / (right - left)
            here = values[index] + fraction*(values[index + 1] - values[index])
            inside = (position >= left) & (position < right)
            result = xp.where(inside, here, result)
        return xp.where(position == float(self.positions[-1]), values[-1], result)

    def antiderivative(self, position, *, xp=None):
        """Exact integral from -inf to position [field unit * m]."""
        xp, (position, values) = real_arrays(position, self.values, xp=xp)
        result = xp.zeros_like(position)
        for index in range(len(self.positions) - 1):
            left, right = float(self.positions[index]), float(self.positions[index + 1])
            if left == right:
                continue
            partial = xp.clip(position - left, 0., right - left)
            result = result + partial*(values[index]
                     + .5*(partial/(right - left))*(values[index + 1] - values[index]))
        return result

    def integral(self, position, *, xp=None):
        """Exact integral from 0 to position [field unit * m]."""
        return self.antiderivative(position, xp=xp) - self.antiderivative(0., xp=xp)

    def __repr__(self):
        return f"SpatialProfile(knot_count={len(self.positions)})"

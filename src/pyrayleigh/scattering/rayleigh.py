"""Stored Rayleigh disorder: one local point coefficient per material cell."""

from dataclasses import dataclass, field
from numbers import Real

import numpy as np

from ..backends import complex_arrays


def _fixed_positions(values, name):
    values = tuple(values)
    if any(isinstance(value, bool) or not isinstance(value, Real) for value in values):
        raise TypeError(f"{name} must contain fixed real Python or NumPy values")
    array = np.array(values, dtype=np.float64)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite")
    array.setflags(write=False)
    return array


@dataclass(frozen=True, eq=False)
class RayleighRealization:
    """One stored sample with explicit cell edges [m] starting at zero.

    coefficients are dimensionless local reflection amplitudes for light
    arriving from the left, without propagation phase or attenuation.
    Their physical domain is finite and |r| <= 1; only numeric dtype and
    shape are checked, preserving native parameters for differentiation.
    positions and edges are fixed host geometry, copied as read-only NumPy
    arrays. There is one scatterer per cell, including either cell edge.
    """

    coefficients: object = field(repr=False)
    positions: object = field(repr=False)
    edges: object = field(repr=False)

    def __post_init__(self):
        xp, (coefficients,) = complex_arrays(self.coefficients)
        if coefficients.ndim != 1 or coefficients.shape[0] == 0:
            raise ValueError("coefficients must be a nonempty one-dimensional array")
        positions = _fixed_positions(self.positions, "positions")
        edges = _fixed_positions(self.edges, "edges")
        if positions.shape != coefficients.shape:
            raise ValueError("positions must have one entry per coefficient")
        if edges.shape != (coefficients.shape[0] + 1,):
            raise ValueError("edges must have one more entry than coefficients")
        if edges[0] != 0. or np.any(np.diff(edges) <= 0.):
            raise ValueError("edges must start at zero and be strictly increasing")
        if np.any(positions < edges[:-1]) or np.any(positions > edges[1:]):
            raise ValueError("each scatterer must lie inside its cell")
        object.__setattr__(self, "coefficients", xp.asarray(coefficients, copy=True))
        object.__setattr__(self, "positions", positions)
        object.__setattr__(self, "edges", edges)

    @property
    def cell_count(self):
        return self.coefficients.shape[0]

    @property
    def extent(self):
        """Last cell edge [m]."""
        return float(self.edges[-1])

    def cell_edges(self):
        return self.edges.copy()

    def cell_lengths(self):
        return np.diff(self.edges)

    def __repr__(self):
        return f"RayleighRealization(cell_count={self.cell_count}, extent={self.extent})"

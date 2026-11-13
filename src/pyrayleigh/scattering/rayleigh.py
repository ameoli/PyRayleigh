"""Stored Rayleigh disorder: one local point coefficient per material cell."""

from dataclasses import dataclass, field
import math
from numbers import Real
import operator

import numpy as np

from ..backends import complex_arrays, real_arrays


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


def _rayleigh_edges(fiber, cell_length):
    if isinstance(cell_length, bool) or not isinstance(cell_length, Real):
        raise TypeError("cell_length must be a fixed real number")
    if not math.isfinite(cell_length) or cell_length <= 0:
        raise ValueError("cell_length must be finite and positive")
    count = math.ceil(fiber.length / cell_length)
    starts = np.arange(count, dtype=np.float64) * cell_length
    return np.append(starts[starts < fiber.length], fiber.length)


def _sample_rayleigh(fiber, edges, normals, *, xp):
    xp, arrays = real_arrays(normals, *(section.rayleigh_coefficient
                                      for section in fiber.sections), xp=xp)
    normals = arrays[0]
    count = len(edges) - 1
    if normals.shape != (count, 2):
        raise ValueError("normals must have shape (cell_count, 2)")
    positions = edges[:-1]
    rate = xp.zeros((count,), dtype=xp.float64)
    for section, value in zip(fiber.sections, arrays[1:]):
        mask = xp.asarray((positions >= section.start) & (positions < section.end))
        rate = xp.where(mask, value, rate)
    widths = xp.asarray(np.diff(edges), dtype=xp.float64)
    amplitude = xp.astype(xp.sqrt(rate*widths/2.), xp.complex128)
    i = xp.asarray(1j, dtype=xp.complex128)
    coefficients = amplitude * (xp.astype(normals[:, 0], xp.complex128)
                                + i*xp.astype(normals[:, 1], xp.complex128))
    return RayleighRealization(coefficients, positions, edges)


def rayleigh_from_normals(fiber, normals, *, cell_length, xp=None):
    """Build a sample from fixed independent N(0, 1) draws of shape (N, 2).

    r_j = sqrt(R(s_j)*ds_j/2) * (normals[j, 0] + i*normals[j, 1]).
    The fixed lattice uses left cell edges, as in PyOFDR. R is sampled at
    those positions, including when a cell crosses a section boundary.
    Keep the draws and cell_length fixed when fitting the section rates;
    derivatives with respect to R require R > 0. No random numbers are drawn.
    """
    edges = _rayleigh_edges(fiber, cell_length)
    return _sample_rayleigh(fiber, edges, normals, xp=xp)


def generate_rayleigh(fiber, seed, *, cell_length, xp=None):
    """Draw a sample on a fixed lattice with the final cell clipped to length.

    Uses a private NumPy PCG64 generator: N real normals, then N imaginary
    normals, as in PyOFDR. The same seed gives the same draws in all backends;
    it is a stream seed, with no interrogator-specific seed derivation here.
    """
    if isinstance(seed, (bool, np.bool_)):
        raise ValueError("seed must be a nonnegative integer")
    try:
        seed = operator.index(seed)
    except TypeError:
        raise ValueError("seed must be a nonnegative integer") from None
    if seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    edges = _rayleigh_edges(fiber, cell_length)
    count = len(edges) - 1
    rng = np.random.Generator(np.random.PCG64(seed))
    normals = np.stack((rng.standard_normal(count), rng.standard_normal(count)), axis=1)
    xp, _ = real_arrays(*(section.rayleigh_coefficient for section in fiber.sections), xp=xp)
    return _sample_rayleigh(fiber, edges, normals, xp=xp)

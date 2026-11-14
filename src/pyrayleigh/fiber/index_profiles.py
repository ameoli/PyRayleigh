"""Stored effective-index perturbations in material coordinates."""

import math
from numbers import Real
import operator

import numpy as np

from ..backends import real_arrays
from ..state.fields import SpatialProfile


def index_segments_profile(segments, *, xp=None):
    """Sum constant delta n values on fixed half-open intervals [start, end)."""
    changes, values = {}, []
    for index, (start, end, delta_index) in enumerate(segments):
        for position in (start, end):
            if isinstance(position, bool) or not isinstance(position, Real):
                raise TypeError("segment positions must be fixed real numbers")
            if not math.isfinite(position):
                raise ValueError("segment positions must be finite")
        start, end = float(start), float(end)
        if start < 0. or end <= start:
            raise ValueError("segments require 0 <= start < end")
        changes.setdefault(start, []).append((index, True))
        changes.setdefault(end, []).append((index, False))
        values.append(delta_index)
    xp, arrays = real_arrays(0., *values, xp=xp)
    zero, amplitudes = arrays[0], arrays[1:]
    if any(value.ndim != 0 for value in amplitudes):
        raise ValueError("segment index changes must be scalars")
    if not changes:
        return SpatialProfile([0., 0.], xp.stack((zero, zero)))
    positions, samples, active = [], [], set()
    value = zero
    for position, events in sorted(changes.items()):
        positions.extend((position, position))
        samples.append(value)
        for index, starts in events:
            if starts:
                active.add(index)
            else:
                active.remove(index)
        value = zero
        for index in sorted(active):
            value = value + amplitudes[index]
        samples.append(value)
    return SpatialProfile(positions, xp.stack(samples))


def _sample_positions(length, sample_spacing):
    for name, value in (("length", length), ("sample_spacing", sample_spacing)):
        if isinstance(value, bool) or not isinstance(value, Real):
            raise TypeError(f"{name} must be a fixed real number")
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    length, sample_spacing = float(length), float(sample_spacing)
    positions = np.arange(math.ceil(length/sample_spacing), dtype=np.float64)*sample_spacing
    return np.append(positions[positions < length], length)


def _index_from_normals(positions, standard_deviation, correlation_length, normals, xp):
    xp, (sigma, correlation_length, normals) = real_arrays(
        standard_deviation, correlation_length, normals, xp=xp)
    if sigma.ndim != 0 or correlation_length.ndim != 0:
        raise ValueError("standard_deviation and correlation_length must be scalars")
    if normals.shape != positions.shape:
        raise ValueError("normals must have one entry per profile knot, including both ends")
    values = [sigma*normals[0]]
    for index, spacing in enumerate(np.diff(positions), start=1):
        exponent = -float(spacing)/correlation_length
        correlation = xp.exp(exponent)
        innovation = sigma*xp.sqrt(-xp.expm1(2.*exponent))
        values.append(correlation*values[-1] + innovation*normals[index])
    return SpatialProfile(positions, xp.stack(values))


def ornstein_uhlenbeck_index_from_normals(length, standard_deviation, correlation_length,
                                         sample_spacing, normals, *, xp=None):
    """Stationary OU index profile from fixed independent N(0, 1) samples.

    Length and spacing [m] are fixed geometry; the final interval is clipped
    to length. Sigma [1] is nonnegative and correlation_length [m] positive.
    Only real dtype and scalar shape are checked for these native parameters.
    The returned profile interpolates linearly between its stored knots.
    """
    positions = _sample_positions(length, sample_spacing)
    return _index_from_normals(positions, standard_deviation, correlation_length, normals, xp)


def ornstein_uhlenbeck_index_profile(length, standard_deviation, correlation_length,
                                    sample_spacing, seed, *, xp=None):
    """Draw one stored OU profile with a private NumPy PCG64 generator.

    Same grid and interpolation as ornstein_uhlenbeck_index_from_normals.
    Seed is a fixed nonnegative integer, without component seed derivation.
    """
    positions = _sample_positions(length, sample_spacing)
    if isinstance(seed, (bool, np.bool_)):
        raise ValueError("seed must be a nonnegative integer")
    try:
        seed = operator.index(seed)
    except TypeError:
        raise ValueError("seed must be a nonnegative integer") from None
    if seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    normals = np.random.Generator(np.random.PCG64(seed)).standard_normal(len(positions))
    xp, _ = real_arrays(standard_deviation, correlation_length, xp=xp)
    return _index_from_normals(positions, standard_deviation, correlation_length, normals, xp)

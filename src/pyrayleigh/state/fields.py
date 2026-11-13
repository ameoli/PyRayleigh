"""Scalar profiles along the fiber and sampled histories in laboratory time."""

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


@dataclass(frozen=True, eq=False)
class TimeHistory:
    """Piecewise linear values in time, held constant outside the samples.

    times [s] are fixed finite Python or NumPy values, strictly increasing.
    There is at least one sample. Values can multiply a spatial profile or
    carry the field's unit when that profile is a unit shape. They must be
    finite; only real dtype and matching shape are checked here.
    """

    times: object = field(repr=False)
    values: object = field(repr=False)

    def __post_init__(self):
        times = tuple(self.times)
        if any(isinstance(value, bool) or not isinstance(value, Real) for value in times):
            raise TypeError("times must contain fixed real Python or NumPy values")
        times = np.array(times, dtype=np.float64)
        if times.size < 1 or not np.all(np.isfinite(times)):
            raise ValueError("times must contain at least one finite sample")
        if np.any(np.diff(times) <= 0):
            raise ValueError("times must be strictly increasing")
        xp, (values,) = real_arrays(self.values)
        if values.shape != times.shape:
            raise ValueError("values must have one entry per time")
        times.setflags(write=False)
        object.__setattr__(self, "times", times)
        object.__setattr__(self, "values", xp.asarray(values, copy=True))

    def value(self, time, *, xp=None):
        """Value at finite laboratory times [s], preserving the query shape."""
        xp, (time, values) = real_arrays(time, self.values, xp=xp)
        result = xp.ones_like(time)*values[0]
        for index in range(len(self.times) - 1):
            left, right = float(self.times[index]), float(self.times[index + 1])
            fraction = xp.clip(time - left, 0., right - left) / (right - left)
            here = values[index] + fraction*(values[index + 1] - values[index])
            inside = (time >= left) & (time < right)
            result = xp.where(inside, here, result)
        return xp.where(time >= float(self.times[-1]), values[-1], result)

    def __repr__(self):
        return f"TimeHistory(sample_count={len(self.times)})"


def _field_inputs(terms, position, time, xp):
    parameters = [position, time]
    for term in terms:
        parameters.append(term.profile.values)
        if term.history is not None:
            parameters.append(term.history.values)
    xp, arrays = real_arrays(*parameters, xp=xp)
    position, time = xp.broadcast_arrays(arrays[0], arrays[1])
    return xp, position, time


@dataclass(frozen=True)
class FieldTerm:
    """One separable profile(s)*history(t); history=None gives a static term."""

    profile: SpatialProfile
    history: TimeHistory | None = None

    def __post_init__(self):
        if not isinstance(self.profile, SpatialProfile):
            raise TypeError("profile must be a SpatialProfile")
        if self.history is not None and not isinstance(self.history, TimeHistory):
            raise TypeError("history must be a TimeHistory or None")

    def _time_factor(self, time, xp):
        if self.history is None:
            return xp.ones_like(time)
        return self.history.value(time, xp=xp)

    def value(self, position, time=0., *, xp=None):
        """Profile value times history, broadcast over position [m] and time [s]."""
        xp, position, time = _field_inputs((self,), position, time, xp)
        return self.profile.value(position, xp=xp) * self._time_factor(time, xp)

    def integral(self, position, time=0., *, xp=None):
        """Exact spatial integral from 0 to position, at the given laboratory time."""
        xp, position, time = _field_inputs((self,), position, time, xp)
        return self.profile.integral(position, xp=xp) * self._time_factor(time, xp)


@dataclass(frozen=True)
class Field:
    """Sum of separable terms of the same physical quantity."""

    terms: tuple = ()

    def __post_init__(self):
        terms = tuple(self.terms)
        if any(not isinstance(term, FieldTerm) for term in terms):
            raise TypeError("terms must contain only FieldTerm objects")
        object.__setattr__(self, "terms", terms)

    @property
    def is_static(self):
        return all(term.history is None for term in self.terms)

    @property
    def is_zero(self):
        """True only for the empty sum, independent of parameter values."""
        return len(self.terms) == 0

    def value(self, position, time=0., *, xp=None):
        """Field value, broadcast over position [m] and laboratory time [s]."""
        xp, position, time = _field_inputs(self.terms, position, time, xp)
        total = xp.zeros_like(position)
        for term in self.terms:
            total = total + term.value(position, time, xp=xp)
        return total

    def integral(self, position, time=0., *, xp=None):
        """Exact spatial integral from 0 to position at fixed laboratory time."""
        xp, position, time = _field_inputs(self.terms, position, time, xp)
        total = xp.zeros_like(position)
        for term in self.terms:
            total = total + term.integral(position, time, xp=xp)
        return total

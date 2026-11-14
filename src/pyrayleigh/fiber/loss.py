"""Loss data stored with the fiber."""

from dataclasses import dataclass, field
import math
from numbers import Real

import numpy as np

from ..backends import real_arrays


def _fixed_grid(values, name, minimum_count):
    values = tuple(values)
    if any(isinstance(value, bool) or not isinstance(value, Real) for value in values):
        raise TypeError(f"{name} must contain fixed real Python or NumPy values")
    values = np.array(values, dtype=np.float64)
    if len(values) < minimum_count or not np.all(np.isfinite(values)):
        raise ValueError(f"{name} needs at least {minimum_count} finite entries")
    if np.any(values <= 0.) or np.any(np.diff(values) <= 0.):
        raise ValueError(f"{name} must be positive and strictly increasing")
    values.setflags(write=False)
    return values


def _interpolate_nonnegative(grid, values, points, xp):
    shape = values.shape[:-1]
    tail = (1,)*points.ndim
    result = xp.asarray(xp.broadcast_to(xp.reshape(values[..., 0], shape + tail),
                                        shape + points.shape), copy=True)
    if len(grid) == 1:
        return result
    points = xp.reshape(points, (1,)*len(shape) + points.shape)
    for index in range(len(grid) - 1):
        low, high = float(grid[index]), float(grid[index + 1])
        fraction = xp.clip((points - low)/(high - low), 0., 1.)
        left = xp.reshape(values[..., index], shape + tail)
        right = xp.reshape(values[..., index + 1], shape + tail)
        positive = (left > 0.) & (right > 0.)
        safe_left = xp.where(positive, left, xp.ones_like(left))
        safe_right = xp.where(positive, right, xp.ones_like(right))
        logarithmic = xp.exp((1. - fraction)*xp.log(safe_left) + fraction*xp.log(safe_right))
        here = xp.where(positive, logarithmic, (1. - fraction)*left + fraction*right)
        result = xp.where((points >= low) & (points < high), here, result)
    last = xp.reshape(values[..., -1], shape + tail)
    result = xp.where(points >= float(grid[-1]), last, result)
    return xp.where(xp.isnan(points), xp.full_like(result, float("nan")), result)


@dataclass(frozen=True, eq=False)
class BendLossTable:
    """Power attenuation alpha [1/m] versus curvature [1/m] and frequency [Hz].

    Grids are fixed, copied as read-only NumPy arrays. There are at least
    two curvature samples and one frequency sample. A single frequency
    means loss independent of frequency. Attenuation has shape (K, F), is
    finite and nonnegative; dtype and shape are checked, keeping native
    float64 copies for differentiation.
    """

    curvatures: object = field(repr=False)
    frequencies: object = field(repr=False)
    attenuation: object = field(repr=False)

    def __post_init__(self):
        curvatures = _fixed_grid(self.curvatures, "curvatures", 2)
        frequencies = _fixed_grid(self.frequencies, "frequencies", 1)
        xp, (attenuation,) = real_arrays(self.attenuation)
        if attenuation.shape != (len(curvatures), len(frequencies)):
            raise ValueError("attenuation must have shape (len(curvatures), len(frequencies))")
        object.__setattr__(self, "curvatures", curvatures)
        object.__setattr__(self, "frequencies", frequencies)
        object.__setattr__(self, "attenuation", xp.asarray(attenuation, copy=True))

    @property
    def shape(self):
        return self.attenuation.shape

    def covers(self, frequency_min, frequency_max):
        """Whether the table covers a fixed, finite positive frequency band [Hz]."""
        for frequency in (frequency_min, frequency_max):
            if isinstance(frequency, bool) or not isinstance(frequency, Real):
                raise TypeError("band limits must be fixed real numbers")
            if not math.isfinite(frequency) or frequency <= 0.:
                raise ValueError("band limits must be finite and positive")
        if frequency_min > frequency_max:
            raise ValueError("frequency_min must not exceed frequency_max")
        return len(self.frequencies) == 1 or bool(
            self.frequencies[0] <= frequency_min and frequency_max <= self.frequencies[-1])

    def by_frequency(self, curvature, *, xp=None):
        """Alpha [1/m] at every table frequency, shape curvature.shape + (F,).

        Uses the absolute curvature [1/m]. Below the first knot the loss is
        zero; above the last knot it is NaN, without host-side value checks.
        Positive pairs interpolate in log(alpha), pairs beside zero linearly.
        Derivatives are intended away from knots and zero-value transitions.
        """
        xp, (curvature, attenuation) = real_arrays(curvature, self.attenuation, xp=xp)
        curvature = xp.abs(curvature)
        columns = [_interpolate_nonnegative(self.curvatures, attenuation[:, index], curvature, xp)
                   for index in range(len(self.frequencies))]
        result = xp.stack(columns, axis=-1)
        curvature = xp.reshape(curvature, curvature.shape + (1,))
        result = xp.where(curvature < float(self.curvatures[0]), xp.zeros_like(result), result)
        return xp.where(curvature <= float(self.curvatures[-1]), result,
                         xp.full_like(result, float("nan")))

    def value(self, curvature, frequency, *, xp=None):
        """Bend alpha [1/m], shape curvature.shape + frequency.shape.

        Frequency interpolation uses the same rule, clamped to the sampled
        band. A one-frequency table is constant over frequency. Queries are
        finite physical inputs; curvature limits follow by_frequency.
        """
        xp, (curvature, frequency, _) = real_arrays(
            curvature, frequency, self.attenuation, xp=xp)
        values = self.by_frequency(curvature, xp=xp)
        return _interpolate_nonnegative(self.frequencies, values, frequency, xp)

    def __repr__(self):
        return f"BendLossTable(shape={self.shape})"

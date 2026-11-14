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

    def __repr__(self):
        return f"BendLossTable(shape={self.shape})"

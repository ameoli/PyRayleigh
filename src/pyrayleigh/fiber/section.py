"""A material interval with uniform fiber properties."""

from dataclasses import dataclass
import math
from numbers import Real

from ..backends import real_arrays
from .optics import GuideOptics


@dataclass(frozen=True)
class Section:
    """Fixed material interval [start, end] [m] of one fiber type.

    attenuation is the power coefficient alpha [1/m].
    rayleigh_coefficient is guided backscattered power per metre [1/m].
    Both coefficients are scalar, finite and nonnegative physical inputs;
    their values are kept native for differentiation, without range checks.
    """

    start: float
    end: float
    optics: GuideOptics
    attenuation: float
    rayleigh_coefficient: float

    def __post_init__(self):
        for value in (self.start, self.end):
            if isinstance(value, bool) or not isinstance(value, Real):
                raise TypeError("section positions must be fixed real numbers")
            if not math.isfinite(value):
                raise ValueError("section positions must be finite")
        if self.start < 0 or self.end <= self.start:
            raise ValueError("section requires 0 <= start < end")
        if not isinstance(self.optics, GuideOptics):
            raise TypeError("optics must be a GuideOptics")
        for name in ("attenuation", "rayleigh_coefficient"):
            _, (value,) = real_arrays(getattr(self, name))
            if value.ndim != 0:
                raise ValueError(f"{name} must be a scalar")

    @property
    def length(self):
        """Material length [m]."""
        return self.end - self.start

    def attenuation_at(self, frequency, *, xp=None):
        """Constant power coefficient [1/m], matching the frequency array shape."""
        xp, (frequency, attenuation) = real_arrays(frequency, self.attenuation, xp=xp)
        return xp.ones_like(frequency) * attenuation

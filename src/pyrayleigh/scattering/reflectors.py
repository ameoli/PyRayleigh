"""Constant two-port coefficients at a fixed material position."""

from dataclasses import dataclass
import math
from numbers import Real

from ..backends import complex_arrays, real_arrays


@dataclass(frozen=True)
class PointReflector:
    """Point element with dimensionless, power-normalized field coefficients.

    position is fixed and in metres. Coefficients must be finite and passive;
    only scalar dtype/shape is checked here to preserve native parameters.
    The matrix is local to this position; propagation is separate.
    """

    position: float
    r_left: complex
    r_right: complex
    t_left_to_right: complex
    t_right_to_left: complex

    def __post_init__(self):
        if isinstance(self.position, bool) or not isinstance(self.position, Real):
            raise TypeError("position must be a fixed real number")
        if not math.isfinite(self.position) or self.position < 0:
            raise ValueError("position must be finite and nonnegative")
        for value in (self.r_left, self.r_right, self.t_left_to_right, self.t_right_to_left):
            _, (array,) = complex_arrays(value)
            if array.ndim != 0:
                raise ValueError("point coefficients must be scalars")

    def scattering_matrix(self, *, xp=None):
        """S with [b_left, b_right] = S @ [a_left, a_right]."""
        xp, (r_left, r_right, t_forward, t_reverse) = complex_arrays(
            self.r_left, self.r_right, self.t_left_to_right, self.t_right_to_left, xp=xp)
        return xp.stack((xp.stack((r_left, t_reverse)), xp.stack((t_forward, r_right))))


def lossless_reflector(position, reflectance, phase=0., *, xp=None):
    """Reciprocal lossless point with power reflectance R and phase [rad].

    R is finite and in [0, 1]. Derivatives are intended for 0 < R < 1;
    the square roots are not smooth at the endpoints.
    """
    xp, (reflectance, phase) = real_arrays(reflectance, phase, xp=xp)
    if reflectance.ndim != 0 or phase.ndim != 0:
        raise ValueError("reflectance and phase must be scalars")
    i = xp.asarray(1j, dtype=xp.complex128)
    r_left = (xp.astype(xp.sqrt(reflectance), xp.complex128)
              * xp.exp(i*xp.astype(phase, xp.complex128)))
    transmission = xp.sqrt(1. - reflectance)
    return PointReflector(position, r_left, -xp.conj(r_left), transmission, transmission)

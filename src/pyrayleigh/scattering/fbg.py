"""Weak FBG spectrum for the localized Born model used by PyOFDR."""

import math
from dataclasses import dataclass, field
from numbers import Real

from ..backends import real_arrays
from ..constants import SPEED_OF_LIGHT


def _sinc(value, xp):
    """sin(pi*x)/(pi*x), with finite derivatives at the origin."""
    angle = math.pi*value
    small = xp.abs(angle) < .1
    series_angle = xp.where(small, angle, xp.zeros_like(angle))
    squared = series_angle*series_angle
    series = 1. + squared*(-1./6. + squared*(1./120. + squared*(-1./5040. + squared/362880.)))
    safe = xp.where(small, xp.ones_like(angle), angle)
    return xp.where(small, series, xp.sin(safe)/safe)


def weak_fbg_reflection(frequency, bragg_wavelength, length, peak_reflectivity, effective_index,
                         *, xp=None):
    """Local field coefficient sqrt(R_max)*sinc(2*n*L*(nu-nu_B)/c).

    Absolute frequencies are in Hz, Bragg wavelength and grating length in
    m. Inputs are finite, nu, wavelength, length and n are positive, and
    0 <= R_max <= 1. The model is intended for weak gratings, R_max << 1.
    Only real dtype is checked. Inputs broadcast and the output is native
    complex128; signed sidelobes retain their phase reversal.

    The supplied Bragg wavelength may already include a prescribed shift.
    Propagation and its losses are applied separately. Derivatives in
    R_max are intended away from zero.
    """
    xp, (frequency, wavelength, length, reflectivity, index) = real_arrays(
        frequency, bragg_wavelength, length, peak_reflectivity, effective_index, xp=xp)
    detuning = frequency - SPEED_OF_LIGHT/wavelength
    argument = 2.*index*length*detuning/SPEED_OF_LIGHT
    reflection = xp.sqrt(reflectivity)*_sinc(argument, xp)
    return xp.astype(reflection, xp.complex128)


@dataclass(frozen=True)
class WeakFBG:
    """Localized weak grating at a fixed material position [m].

    Bragg wavelength and length are in m; peak_reflectivity is a small
    power fraction. These are native scalars with the spectrum function's
    preconditions. Length sets the bandwidth, rather than a layout interval.
    As in PyOFDR's Born model, the grating adds reflection and no depletion.
    """

    position: float
    bragg_wavelength: object
    length: object
    peak_reflectivity: object
    t_left_to_right: float = field(default=1., init=False)
    t_right_to_left: float = field(default=1., init=False)

    def __post_init__(self):
        if isinstance(self.position, bool) or not isinstance(self.position, Real):
            raise TypeError("position must be a fixed real number")
        if not math.isfinite(self.position) or self.position < 0.:
            raise ValueError("position must be finite and nonnegative")
        for value in self._parameters():
            _, (value,) = real_arrays(value)
            if value.ndim != 0:
                raise ValueError("grating parameters must be scalars")

    def _parameters(self):
        return self.bragg_wavelength, self.length, self.peak_reflectivity

    def reflection(self, frequency, effective_index, relative_shift=0., *, xp=None):
        """Evaluate from the stored resonance; invalid shifted wavelengths give NaN."""
        xp, (frequency, index, wavelength, length, reflectivity, shift) = real_arrays(
            frequency, effective_index, *self._parameters(), relative_shift, xp=xp)
        wavelength = wavelength*(1. + shift)
        valid = xp.isfinite(wavelength) & (wavelength > 0.)
        safe = xp.where(valid, wavelength, xp.ones_like(wavelength))
        reflection = weak_fbg_reflection(frequency, safe, length, reflectivity, index, xp=xp)
        return xp.where(valid, reflection, xp.full_like(reflection, complex(float("nan"))))

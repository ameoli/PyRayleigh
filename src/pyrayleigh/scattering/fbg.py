"""Weak FBG spectrum for the localized Born model used by PyOFDR."""

import math

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

"""Conversions to SI units. Attenuation is a power coefficient, in 1/m."""

import math

import array_api_compat as apc
import array_api_compat.numpy as numpy_api

from .backends import array_namespace, asarray
from .constants import SPEED_OF_LIGHT

PS_PER_NM_KM = 1e-6  # s/m^2
_LN10 = math.log(10.)


def _real_inputs(*values, xp=None):
    if xp is None:
        arrays = [value for value in values if apc.is_array_api_obj(value)]
        xp = array_namespace(*arrays) if arrays else numpy_api
    arrays = tuple(asarray(value, xp=xp) for value in values)
    if any(xp.isdtype(value.dtype, "complex floating") for value in arrays):
        raise TypeError("unit conversions require real values")
    return xp, arrays


def db_to_linear(value_db, *, xp=None):
    """Power ratio from dB."""
    xp, (value,) = _real_inputs(value_db, xp=xp)
    return xp.exp(value * (_LN10 / 10.))


def linear_to_db(value, *, xp=None):
    """dB from a positive power ratio; zero corresponds to -inf."""
    xp, (value,) = _real_inputs(value, xp=xp)
    return 10. * xp.log10(value)


def attenuation_from_db_per_km(attenuation_db_per_km, *, xp=None):
    """Power loss alpha [1/m], with P(z) = P(0) exp(-alpha*z)."""
    xp, (value,) = _real_inputs(attenuation_db_per_km, xp=xp)
    return value * (_LN10 / 10000.)


def attenuation_to_db_per_km(attenuation, *, xp=None):
    """Loss [dB/km] from the power attenuation coefficient [1/m]."""
    xp, (value,) = _real_inputs(attenuation, xp=xp)
    return value * (10000. / _LN10)


def wavelength_to_frequency(wavelength, *, xp=None):
    """Absolute optical frequency [Hz] from positive vacuum wavelength [m]."""
    xp, (value,) = _real_inputs(wavelength, xp=xp)
    return SPEED_OF_LIGHT / value


def frequency_to_wavelength(frequency, *, xp=None):
    """Vacuum wavelength [m] from positive absolute optical frequency [Hz]."""
    xp, (value,) = _real_inputs(frequency, xp=xp)
    return SPEED_OF_LIGHT / value


def dispersion_to_gvd_at_wavelength(dispersion, wavelength, *, xp=None):
    """beta2 [s^2/m] from D [s/m^2] and positive vacuum wavelength [m]."""
    xp, (dispersion, wavelength) = _real_inputs(dispersion, wavelength, xp=xp)
    return -wavelength**2 * dispersion / (2. * xp.pi * SPEED_OF_LIGHT)

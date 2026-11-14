"""Empirical macrobend loss, with the same parameters as PyOFDR."""

import math
from numbers import Real

import numpy as np

from ..backends import real_arrays
from ..constants import SPEED_OF_LIGHT
from ..fiber.loss import BendLossTable, _fixed_grid

NOMINAL_FREQUENCY = SPEED_OF_LIGHT/1550e-9
DEFAULT_TABLE_RADII = np.geomspace(100e-3, 1e-3, 200)
DEFAULT_TABLE_RADII.setflags(write=False)


def bend_length(bend_radius, turns, *, xp=None):
    """Arc length [m] for positive radius [m] and nonnegative turns."""
    xp, (radius, turns) = real_arrays(bend_radius, turns, xp=xp)
    return 2.*math.pi*radius*turns


def attenuation_from_loss_db(loss_db, length, *, xp=None):
    """Uniform power alpha [1/m] for loss_db >= 0 over length > 0 [m]."""
    xp, (loss, length) = real_arrays(loss_db, length, xp=xp)
    return loss*(math.log(10.)/10.)/length


def loss_db_from_attenuation(attenuation, length, *, xp=None):
    """Power loss [dB] for alpha >= 0 [1/m] over length >= 0 [m]."""
    xp, (alpha, length) = real_arrays(attenuation, length, xp=xp)
    return alpha*length*(10./math.log(10.))


def exponential_bend_loss_db(radius, turns, loss_per_turn_db=100., critical_radius=5e-3,
                              *, xp=None):
    """Total one-way loss [dB]: turns * A * exp(-radius / critical_radius).

    Radii [m] are positive; turns and A [dB/turn] are nonnegative. Inputs
    are finite and broadcast normally. Only real dtype is checked, keeping
    native parameters differentiable. This is PyOFDR's empirical model.
    """
    xp, (radius, turns, amplitude, critical) = real_arrays(
        radius, turns, loss_per_turn_db, critical_radius, xp=xp)
    return turns*amplitude*xp.exp(-radius/critical)


def exponential_bend_loss_table(bend_radii=DEFAULT_TABLE_RADII, loss_per_turn_db=100.,
                                 critical_radius=5e-3, frequency=NOMINAL_FREQUENCY, *, xp=None):
    """Sample the empirical law into a frequency-independent BendLossTable.

    Radii [m] are fixed, positive and distinct, in any order (at least two).
    Frequency [Hz] is a fixed positive label. A >= 0 and R_c > 0 are native
    finite scalars; only their real dtype and scalar shape are checked.
    Zero and underflowed losses stay zero.

    Alpha uses one physical turn of length 2*pi*R. Values are exact at the
    sampled radii; between them log(alpha) is interpolated in curvature.
    For PyOFDR's arbitrary [start, end] spans, convert the total dB with
    attenuation_from_loss_db(total, end - start) instead.
    """
    radii = tuple(bend_radii)
    if any(isinstance(value, bool) or not isinstance(value, Real) for value in radii):
        raise TypeError("bend_radii must contain fixed real Python or NumPy values")
    radii = _fixed_grid(sorted(radii), "bend_radii", 2)[::-1]
    xp, (amplitude, critical) = real_arrays(loss_per_turn_db, critical_radius, xp=xp)
    if amplitude.ndim != 0 or critical.ndim != 0:
        raise ValueError("loss_per_turn_db and critical_radius must be scalars")
    native_radii = xp.asarray(radii.tolist(), dtype=xp.float64)
    losses = exponential_bend_loss_db(native_radii, 1., amplitude, critical, xp=xp)
    lengths = bend_length(native_radii, 1., xp=xp)
    alpha = attenuation_from_loss_db(losses, lengths, xp=xp)
    return BendLossTable(1./radii, [frequency], alpha[:, None])

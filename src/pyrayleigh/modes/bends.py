"""Macrobend power loss, with the same conventions as PyOFDR."""

import math
from numbers import Integral, Real

import numpy as np

from ..backends import real_arrays
from ..constants import SPEED_OF_LIGHT
from ..fiber.loss import BendLossTable, _fixed_grid, _interpolate_nonnegative
from .step_index import _cladding_bessel, solve_lp01_mode

NOMINAL_FREQUENCY = SPEED_OF_LIGHT/1550e-9
DEFAULT_TABLE_RADII = np.geomspace(100e-3, 1e-3, 200)
DEFAULT_TABLE_RADII.setflags(write=False)


def _sorted_radii(radii):
    radii = tuple(radii)
    if any(isinstance(value, bool) or not isinstance(value, Real) for value in radii):
        raise TypeError("bend_radii must contain fixed real Python or NumPy values")
    order = sorted(range(len(radii)), key=radii.__getitem__)
    return _fixed_grid([radii[index] for index in order], "bend_radii", 2), order


def _turn_samples(radii, losses, xp):
    radii, order = _sorted_radii(radii)
    xp, (losses,) = real_arrays(losses, xp=xp)
    if losses.shape != radii.shape:
        raise ValueError("give one loss per turn for each radius")
    return xp, radii, xp.stack([losses[index] for index in order])


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


def per_turn_table(radii, loss_per_turn_db, frequency=NOMINAL_FREQUENCY, *, xp=None):
    """Convert fixed radius samples [m] and native losses [dB/turn] to alpha.

    Radii are distinct and positive, in any order (at least two). Losses
    are a finite nonnegative vector, one per radius. Only real dtype and
    shape are checked. Alpha uses the physical length 2*pi*R of one turn;
    zero and underflowed losses stay zero. Frequency is a fixed Hz label.
    """
    xp, radii, losses = _turn_samples(radii, loss_per_turn_db, xp)
    lengths = bend_length(xp.asarray(radii.tolist(), dtype=xp.float64), 1., xp=xp)
    alpha = attenuation_from_loss_db(losses, lengths, xp=xp)
    return BendLossTable(1./radii[::-1], [frequency], xp.flip(alpha, axis=0)[:, None])


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
    radii, _ = _sorted_radii(bend_radii)
    xp, (amplitude, critical) = real_arrays(loss_per_turn_db, critical_radius, xp=xp)
    if amplitude.ndim != 0 or critical.ndim != 0:
        raise ValueError("loss_per_turn_db and critical_radius must be scalars")
    native_radii = xp.asarray(radii.tolist(), dtype=xp.float64)
    losses = exponential_bend_loss_db(native_radii, 1., amplitude, critical, xp=xp)
    return per_turn_table(radii, losses, frequency, xp=xp)


def tabulated_bend_loss_db(radius, turns, table_radii, table_loss_per_turn_db, *, xp=None):
    """Total one-way dB loss, log-linear in radius as in PyOFDR.

    Table radii [m] are fixed, positive and distinct, in any order. Native
    losses [dB/turn] are finite and strictly positive, one per radius;
    only real dtype and shape are checked. Radius > 0 and turns >= 0
    broadcast normally. End samples clamp outside the radius range,
    without host-side range warnings, so JIT queries stay native.
    """
    xp, (radius, turns, losses) = real_arrays(radius, turns, table_loss_per_turn_db, xp=xp)
    xp, radii, losses = _turn_samples(table_radii, losses, xp)
    return turns*_interpolate_nonnegative(radii, losses, radius, xp)


def tabulated_bend_loss_table(table_radii, table_loss_per_turn_db, points_per_interval=32,
                              frequency=NOMINAL_FREQUENCY, *, xp=None):
    """Resample the radius log-loss law into a frequency-independent table.

    Same inputs as tabulated_bend_loss_db. Each radius interval has a fixed
    positive integer number of subdivisions; original samples are retained.
    Between resampled radii, curvature interpolation is an approximation.
    The resulting table uses zero loss above the largest radius and NaN
    below the smallest, following BendLossTable's curvature limits.
    """
    if isinstance(points_per_interval, bool) or not isinstance(points_per_interval, Integral):
        raise TypeError("points_per_interval must be a fixed integer")
    if points_per_interval < 1:
        raise ValueError("points_per_interval must be positive")
    xp, radii, losses = _turn_samples(table_radii, table_loss_per_turn_db, xp)
    dense = np.unique(np.concatenate([
        np.linspace(low, high, int(points_per_interval) + 1)[:-1]
        for low, high in zip(radii[:-1], radii[1:])] + [radii[-1:]]))
    query = xp.asarray(dense.tolist(), dtype=xp.float64)
    dense_losses = _interpolate_nonnegative(radii, losses, query, xp)
    return per_turn_table(dense, dense_losses, frequency, xp=xp)


def _marcuse_log_attenuation(frequency, core_radius, core_index, cladding_index,
                              bend_radius, radius_factor, xp):
    mode = solve_lp01_mode(frequency, core_radius, core_index, cladding_index, xp=xp)
    beta = mode.propagation_constant
    effective_radius = bend_radius*radius_factor
    valid = (xp.isfinite(mode.U) & xp.isfinite(beta) & xp.isfinite(bend_radius)
             & xp.isfinite(radius_factor) & xp.isfinite(effective_radius)
             & (radius_factor > 0.) & (bend_radius > mode.core_radius)
             & (effective_radius > mode.core_radius))
    def safe(value):
        return xp.where(valid, value, xp.ones_like(value))
    log_a = xp.log(safe(mode.core_radius))
    log_w = xp.log(safe(mode.W))
    log_radius = xp.log(safe(effective_radius))
    _, wk1 = _cladding_bessel(log_w, xp)
    exponent = xp.exp(math.log(2./3.) + 3.*(log_w - log_a) + log_radius - 2.*xp.log(safe(beta)))
    # Combine the W powers before taking logs of K1 at weak confinement.
    log_rate = (.5*math.log(math.pi) - math.log(2.) + 2.*xp.log(safe(mode.U))
                 + .5*(log_w - log_a - log_radius) - 2.*xp.log(safe(mode.V))
                 - 2.*xp.log(wk1) - exponent)
    return xp.where(valid, log_rate, xp.full_like(log_rate, float("nan")))


def marcuse_bend_attenuation(frequency, core_radius, core_index, cladding_index,
                             bend_radius, radius_factor=1.28, *, xp=None):
    """Marcuse LP01 power alpha [1/m] along a physical bend radius [m].

    Frequency is in Hz; the guide uses solve_lp01_mode's supported range.
    Finite R > core_radius and radius_factor > 0 with R_eff > core_radius
    are required. R_eff = R*radius_factor affects the radiation rate only.
    All inputs broadcast and remain native; invalid geometry, unsupported
    modes and overflowing rates give NaN. Numerical underflow gives zero.
    """
    xp, values = real_arrays(frequency, core_radius, core_index, cladding_index,
                              bend_radius, radius_factor, xp=xp)
    log_rate = _marcuse_log_attenuation(*values, xp)
    rate = xp.exp(log_rate)
    return xp.where(xp.isfinite(rate), rate, xp.full_like(rate, float("nan")))


def marcuse_bend_loss_db(radius, turns, wavelength, core_radius, core_index, cladding_index,
                         radius_correction=1.28, *, xp=None):
    """Total one-way Marcuse LP01 loss [dB], with PyOFDR's argument convention.

    Wavelength and radii are in m. Turns and wavelength must be finite and
    strictly positive. The physical arc is 2*pi*radius*turns, without the
    radius correction. Domain limits and NaN handling follow the rate
    function. Total loss is evaluated in log form before exponentiating.
    """
    xp, (radius, turns, wavelength, a, n1, n2, correction) = real_arrays(
        radius, turns, wavelength, core_radius, core_index, cladding_index, radius_correction, xp=xp)
    valid = xp.isfinite(wavelength) & (wavelength > 0.) & xp.isfinite(turns) & (turns > 0.)
    frequency = SPEED_OF_LIGHT/xp.where(valid, wavelength, xp.ones_like(wavelength))
    log_rate = _marcuse_log_attenuation(frequency, a, n1, n2, radius, correction, xp)
    safe_radius = xp.where(xp.isfinite(radius) & (radius > 0.), radius, xp.ones_like(radius))
    safe_turns = xp.where(valid, turns, xp.ones_like(turns))
    log_loss = (log_rate + math.log(10./math.log(10.)) + math.log(2.*math.pi)
                 + xp.log(safe_radius) + xp.log(safe_turns))
    loss = xp.exp(log_loss)
    return xp.where(valid & xp.isfinite(loss), loss, xp.full_like(loss, float("nan")))


def marcuse_bend_loss_table(frequencies, core_radius, core_index, cladding_index,
                             bend_radii=DEFAULT_TABLE_RADII, radius_factor=1.28, *, xp=None):
    """Sample Marcuse LP01 power rates into a BendLossTable.

    Frequencies [Hz] are fixed, positive and strictly increasing (at least
    one). Radii [m] are fixed, positive and distinct, in any order (at least
    two). Core radius, both indices and radius_factor are native scalars.
    The returned (curvature, frequency) rates stay native float64, including
    underflowed zeros and NaN samples outside the model's supported range.

    A single frequency freezes the model there, as at PyOFDR's centre
    wavelength. Rates are exact at the sampled pairs; interpolation between
    them follows BendLossTable. They apply over physical bent lengths;
    arbitrary PyOFDR spans use marcuse_bend_loss_db and attenuation_from_loss_db.
    """
    frequencies = _fixed_grid(frequencies, "frequencies", 1)
    radii, _ = _sorted_radii(bend_radii)
    radii = radii[::-1]
    xp, parameters = real_arrays(core_radius, core_index, cladding_index, radius_factor, xp=xp)
    if any(value.ndim != 0 for value in parameters):
        raise ValueError("core radius, indices and radius_factor must be scalars")
    frequency = xp.asarray(frequencies.tolist(), dtype=xp.float64)
    radius = xp.asarray(radii.tolist(), dtype=xp.float64)
    a, n1, n2, correction = parameters
    rates = marcuse_bend_attenuation(frequency[None, :], a, n1, n2, radius[:, None], correction, xp=xp)
    return BendLossTable(1./radii, frequencies, rates)

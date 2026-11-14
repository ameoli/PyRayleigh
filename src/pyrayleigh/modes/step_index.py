"""Scalar LP01 kernels for the single-mode range used by PyOFDR."""

import math
from dataclasses import dataclass, field

from ..backends import fixed_loop, real_arrays
from ..constants import SPEED_OF_LIGHT

LP11_CUTOFF = 2.404825557695773
_EULER_GAMMA = .5772156649015328606


def normalized_frequency(frequency, core_radius, core_index, cladding_index, *, xp=None):
    """V = (2*pi*nu*a/c)*sqrt(n1**2 - n2**2), with normal broadcasting.

    Frequency [Hz] and core radius [m] are positive, n1 > n2 > 0. Inputs
    are finite; only real dtype is checked, retaining native float64 data.
    """
    xp, (frequency, radius, n1, n2) = real_arrays(
        frequency, core_radius, core_index, cladding_index, xp=xp)
    return (2.*math.pi*frequency*radius/SPEED_OF_LIGHT)*xp.sqrt((n1 - n2)*(n1 + n2))


def _core_bessel(u_squared, xp):
    """J0(U) and U*J1(U), for 0 <= U <= LP11_CUTOFF, without sqrt(U**2)."""
    term = xp.ones_like(u_squared)
    j0, uj1 = term, xp.zeros_like(term)
    for order in range(1, 21):
        term = term*(-.25*u_squared)/(order*order)
        j0 = j0 + term
        uj1 = uj1 - 2.*order*term
    return j0, uj1


def _cladding_bessel(log_w, xp):
    """K0(W) and W*K1(W), tested for -300 <= log(W) <= log(LP11_CUTOFF)."""
    # Power series: DLMF 10.31.2, with W*K1 = -d(K0)/d(log W).
    squared = .25*xp.exp(2.*log_w)
    term = xp.ones_like(log_w)
    i0, wi1 = term, xp.zeros_like(term)
    series, derivative = xp.zeros_like(term), xp.zeros_like(term)
    harmonic = 0.
    for order in range(1, 21):
        harmonic = harmonic + 1./order
        term = term*squared/(order*order)
        i0 = i0 + term
        wi1 = wi1 + 2.*order*term
        series = series + harmonic*term
        derivative = derivative + 2.*order*harmonic*term
    logarithm = log_w - math.log(2.) + _EULER_GAMMA
    return -logarithm*i0 + series, i0 + logarithm*wi1 - derivative


def _lp01_characteristic(log_w, v, *, xp=None):
    """U*J1(U)*K0(W) - W*K1(W)*J0(U), using U**2 = V**2 - W**2.

    Dimensionless finite inputs broadcast normally. The single-mode range
    is 0 < V < LP11_CUTOFF, -300 <= log(W) <= log(V). No host-side range
    checks or root search are performed here. The products avoid division
    by J0 near cutoff and by W at weak confinement.
    """
    xp, (log_w, v) = real_arrays(log_w, v, xp=xp)
    return _lp01_value_slope(log_w, v, xp)[0]


def _lp01_value_slope(log_w, v, xp):
    w = xp.exp(log_w)
    squared = (v - w)*(v + w)
    j0, uj1 = _core_bessel(squared, xp)
    k0, wk1 = _cladding_bessel(log_w, xp)
    safe = xp.where(squared == 0., xp.ones_like(squared), squared)
    ratio = xp.where(squared == 0., xp.full_like(squared, .5), uj1/safe)
    return uj1*k0 - wk1*j0, -wk1*v*v*ratio


def _lp01_log_w(v, xp):
    supported = xp.isfinite(v) & (v > 0.) & (v < LP11_CUTOFF)
    safe_v = xp.where(supported, v, xp.ones_like(v))
    supported = supported & (_lp01_value_slope(xp.full_like(v, -300.), safe_v, xp)[0] > 0.)
    safe_v = xp.where(supported, safe_v, xp.ones_like(v))
    low, high = xp.full_like(v, -300.), xp.log(safe_v)
    j0, vj1 = _core_bessel(safe_v*safe_v, xp)
    log_w = xp.clip(math.log(2.) - _EULER_GAMMA - j0/vj1, low, high)
    def step(_, state):
        low, high, log_w = state
        value, slope = _lp01_value_slope(log_w, safe_v, xp)
        low = xp.where(value > 0., log_w, low)
        high = xp.where(value < 0., log_w, high)
        proposal = log_w - value/slope
        inside = (proposal >= low) & (proposal <= high)
        return low, high, xp.where(inside, proposal, .5*(low + high))
    _, _, log_w = fixed_loop(12, step, (low, high, log_w), xp=xp)
    # Full Newton steps retain first and second derivatives after bracketing.
    for _ in range(2):
        value, slope = _lp01_value_slope(log_w, safe_v, xp)
        log_w = log_w - value/slope
    value, slope = _lp01_value_slope(log_w, safe_v, xp)
    converged = (xp.abs(value/slope) <= 2e-13*(1. + xp.abs(log_w)))
    converged = converged & (log_w >= -300.) & (log_w <= xp.log(safe_v))
    return xp.where(supported & converged, log_w, xp.full_like(log_w, float("nan")))


@dataclass(frozen=True, eq=False)
class StepIndexMode:
    """LP01 solution; every numerical field is a native float64 broadcast array.

    Frequency is in Hz, core_radius in m. V, U, W, b=W**2/V**2 and both
    phase indices are dimensionless. Arrays own their data when returned
    by solve_lp01_mode.
    """

    frequency: object
    core_radius: object
    core_index: object
    cladding_index: object
    V: object
    U: object
    W: object
    b: object
    effective_index: object
    name: str = field(default="LP01", init=False)
    azimuthal_order: int = field(default=0, init=False)
    radial_order: int = field(default=1, init=False)

    @property
    def propagation_constant(self):
        """Beta [rad/m]."""
        return 2.*math.pi*self.frequency*self.effective_index/SPEED_OF_LIGHT


def solve_lp01_mode(frequency, core_radius, core_index, cladding_index, *, xp=None):
    """Solve the weakly guiding, single-mode step-index problem as in PyOFDR.

    Finite nu > 0 [Hz], a > 0 [m], n1 > n2 >= 1 and Delta <= .01 are
    required. All inputs broadcast and stay native. Derived fields are NaN
    outside 0 < V < LP11_CUTOFF, for unresolved log(W) < -300, or when the
    fixed Newton iteration does not converge. Only real dtype is checked
    on the host. Differentiation is intended inside the supported range.
    """
    xp, inputs = real_arrays(frequency, core_radius, core_index, cladding_index, xp=xp)
    frequency, radius, n1, n2 = xp.broadcast_arrays(*inputs)
    valid = (xp.isfinite(frequency) & xp.isfinite(radius) & xp.isfinite(n1) & xp.isfinite(n2)
             & (frequency > 0.) & (radius > 0.) & (n1 > n2) & (n2 >= 1.))
    safe_n1 = xp.where(valid, n1, xp.full_like(n1, 1.5))
    safe_n2 = xp.where(valid, n2, xp.full_like(n2, 1.4))
    delta = .5*((safe_n1 - safe_n2)/safe_n1)*(1. + safe_n2/safe_n1)
    valid = valid & (delta <= .01)
    v = normalized_frequency(xp.where(valid, frequency, xp.ones_like(frequency)),
                              xp.where(valid, radius, xp.ones_like(radius)),
                              xp.where(valid, safe_n1, xp.full_like(n1, 1.5)),
                              xp.where(valid, safe_n2, xp.full_like(n2, 1.4)), xp=xp)
    log_w = _lp01_log_w(xp.where(valid, v, xp.ones_like(v)), xp)
    valid = valid & xp.isfinite(log_w)
    safe_v = xp.where(valid, v, xp.ones_like(v))
    w = xp.exp(xp.where(valid, log_w, xp.zeros_like(log_w)))
    u = xp.sqrt((safe_v - w)*(safe_v + w))
    b = (w/safe_v)**2
    safe_n1 = xp.where(valid, n1, xp.full_like(n1, 1.5))
    safe_n2 = xp.where(valid, n2, xp.full_like(n2, 1.4))
    effective_index = xp.sqrt(safe_n2**2 + b*(safe_n1 - safe_n2)*(safe_n1 + safe_n2))
    derived = [xp.where(valid, value, xp.full_like(value, float("nan")))
               for value in (v, u, w, b, effective_index)]
    names = ("frequency", "core_radius", "core_index", "cladding_index", "V", "U", "W", "b",
             "effective_index")
    values = (frequency, radius, n1, n2, *derived)
    return StepIndexMode(**{name: xp.asarray(value, copy=True) for name, value in zip(names, values)})

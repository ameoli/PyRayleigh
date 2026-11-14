"""Scalar LP01 kernels for the single-mode range used by PyOFDR."""

import math

from ..backends import real_arrays
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
    w = xp.exp(log_w)
    j0, uj1 = _core_bessel((v - w)*(v + w), xp)
    k0, wk1 = _cladding_bessel(log_w, xp)
    return uj1*k0 - wk1*j0

"""Uniform, lossless coupled-mode section in the Bragg envelope frame."""

import array_api_compat as apc
import numpy as np

from ..backends import complex_arrays, real_arrays


def uniform_bragg_two_port(detuning, coupling, length):
    """NumPy CMT solution, with detuning and complex coupling in 1/m.

    length [m] is a finite nonnegative scalar; the other inputs broadcast.
    delta=beta-beta_B; the grating phase is carried by kappa.
    The envelope equations are A'=-i*delta*A-i*kappa*B and
    B'=i*conj(kappa)*A+i*delta*B, with A travelling right and B left.
    Both ports use this common rotating frame: kappa=0 gives
    t=exp(-i*delta*length). Optical carrier phase is not included.
    No background loss, material mapping or imposed state is applied here.
    """
    from ..propagation.cascade import TwoPort

    xp, _ = complex_arrays(detuning, coupling, length)
    if not apc.is_numpy_namespace(xp):
        raise NotImplementedError('uniform_bragg_two_port currently supports NumPy only')
    _, (delta, width) = real_arrays(detuning, length, xp=xp)
    _, (kappa,) = complex_arrays(coupling, xp=xp)
    if width.ndim != 0 or not np.isfinite(width) or width < 0.:
        raise ValueError('length must be a finite nonnegative scalar')
    if not np.all(np.isfinite(delta)) or not np.all(np.isfinite(kappa)):
        raise ValueError('detuning and coupling must be finite')

    with np.errstate(over='raise', invalid='raise', divide='raise', under='ignore'):
        d, k = np.broadcast_arrays(delta*width, kappa*width)
        magnitude = np.abs(k)
        squared = (magnitude-np.abs(d))*(magnitude+np.abs(d))
        u = np.sqrt(np.abs(squared))
        stop = squared >= 0.
        c = np.array(np.cos(u))
        s = np.array(np.sinc(u/np.pi))
        numerator = np.ones_like(u)

        # Divide out cosh in the stop band, avoiding overflow for strong gratings.
        inside = u[stop]
        ratio = np.ones_like(inside)
        np.divide(np.tanh(inside), inside, out=ratio, where=inside != 0.)
        c[stop], s[stop] = 1., ratio
        decay = np.exp(-inside)
        numerator[stop] = 2.*decay/(1.+decay*decay)

        denominator = c+1j*d*s
        transmission = numerator/denominator
        result = TwoPort(-1j*np.conj(k)*s/denominator, -1j*k*s/denominator,
            transmission, transmission)
    result.validate_passivity()
    return result

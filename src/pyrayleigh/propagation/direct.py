"""Single-reflection paths with the same local transmissions as the exact model."""

import array_api_compat as apc
import numpy as np

from ..backends import real_arrays
from .exact import _response_blocks
from .cascade import _TOLERANCE


def direct_response_consistent(fiber, probe, state=None, *, loss_convention,
                               denominator_tolerance=_TOLERANCE, xp=None):
    """Return one-reflection returns and the no-reflection transmitted field.

    One scalar core and a static or frozen state. Background loss, point
    coefficients and stored Rayleigh depletion are shared with exact_response.
    A Bragg interval is one block, including its internal CMT, point and
    Rayleigh scattering. Internal paths remain in the direct block response.
    denominator_tolerance also applies to its internal piecewise cascade.
    This path truncation need not conserve power for strong reflectors.
    Fields stay in the selected native namespace. NumPy keeps numerical
    exceptions; invalid native model parameters or arithmetic give NaN fields.
    With JAX, extract the response fields inside a jit wrapper.
    """
    from ..response import Response

    xp, blocks, valid = _response_blocks(fiber, probe, state, loss_convention, denominator_tolerance, xp)
    is_numpy = apc.is_numpy_namespace(xp)
    _, (frequency,) = real_arrays(probe.frequency, xp=xp)
    reflection = xp.zeros_like(frequency, dtype=xp.complex128)
    forward = xp.ones_like(reflection)
    backward =  xp.ones_like(reflection)
    policy = 'raise' if is_numpy else 'ignore'
    with np.errstate(over=policy, invalid=policy, divide=policy, under='ignore'):
        for _, block in blocks:
            reflection = reflection + backward*block.r_left*forward
            forward = forward*block.t_left_to_right
            backward = backward*block.t_right_to_left
            if is_numpy and np.all(forward == 0.):
                break
    if not is_numpy:
        valid = valid & xp.isfinite(reflection) & xp.isfinite(forward)
        reflection = xp.where(valid, reflection, xp.full_like(reflection, float('nan')))
        forward = xp.where(valid, forward, xp.full_like(forward, float('nan')))
    return Response(probe, reflection=reflection, transmission=forward)

"""Single-reflection paths with the same local transmissions as the exact model."""

import numpy as np

from .exact import _static_blocks
from .cascade import _TOLERANCE


def direct_response_consistent(fiber, probe, state=None, *, loss_convention,
                               denominator_tolerance=_TOLERANCE):
    """Return one-reflection returns and the no-reflection transmitted field.

    NumPy, one scalar core and a static state only. Background loss, point
    coefficients and stored Rayleigh depletion are shared with exact_response.
    A Bragg interval is one block, including its internal CMT reflections.
    denominator_tolerance also applies to its internal piecewise cascade.
    This path truncation need not conserve power for strong reflectors.
    """
    from ..response import Response

    blocks = _static_blocks(fiber, probe, state, loss_convention, denominator_tolerance)
    reflection = np.zeros_like(probe.frequency, dtype=np.complex128)
    forward = np.ones_like(reflection)
    backward =  np.ones_like(reflection)
    with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
        for _, block in blocks:
            reflection = reflection + backward*block.r_left*forward
            forward = forward*block.t_left_to_right
            backward = backward*block.t_right_to_left
            if np.all(forward == 0.):
                break
    return Response(probe, reflection=reflection, transmission=forward)

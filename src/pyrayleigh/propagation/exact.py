"""Scalar static NumPy reference, using stored point and Rayleigh two-ports."""

import array_api_compat as apc
import numpy as np

from ..backends import complex_arrays
from ..fiber import Fiber, MultiCoreFiber
from ..probe import Probe
from ..scattering import PointReflector
from ..state import FiberState
from .cascade import NumericalResonanceError, TwoPort, _TOLERANCE, star
from .phase import optical_path


def exact_response(fiber, probe, state=None, *, loss_convention,
                   denominator_tolerance=_TOLERANCE):
    """Left reflection and transmission, with all orders of discrete scattering.

    Section attenuation must be declared as background loss. Only stored
    Rayleigh samples are used; no random draws or grid changes are made.
    A single Fiber, static state and NumPy parameters are supported.
    """
    from ..response import Response

    if not isinstance(loss_convention, str) or loss_convention != "background":
        raise ValueError("exact_response requires loss_convention='background'")
    if isinstance(fiber, MultiCoreFiber):
        raise NotImplementedError("exact_response currently supports one scalar core")
    if not isinstance(fiber, Fiber) or not isinstance(probe, Probe):
        raise TypeError("exact_response requires a Fiber and a Probe")
    state =  FiberState() if state is None else state
    if not isinstance(state, FiberState):
        raise TypeError("state must be a FiberState")
    if not state.is_static:
        raise NotImplementedError("exact_response requires a static or frozen state")
    if any(not isinstance(point, PointReflector) for point in fiber.elements):
        raise NotImplementedError("exact_response currently supports PointReflector elements only")

    path = optical_path(fiber, state)
    parameters = path._parameters(probe.frequency, extra=path.loss._parameters())
    if probe.time is not None:
        parameters.append(probe.time)
    for section in fiber.sections:
        parameters.extend((section.rayleigh_coefficient, section.optics.thermal_expansion))
    for point in fiber.elements:
        parameters.extend((point.r_left, point.r_right, point.t_left_to_right, point.t_right_to_left))
    if fiber.rayleigh is not None:
        parameters.append(fiber.rayleigh.coefficients)
    xp, arrays = complex_arrays(*parameters)
    if not apc.is_numpy_namespace(xp):
        raise NotImplementedError("exact_response currently supports NumPy only")
    if any(not np.all(np.isfinite(value)) for value in arrays):
        raise ValueError("exact_response parameters must be finite")
    frequency = np.asarray(probe.frequency)
    if np.any(frequency <= 0.) or np.any(frequency < fiber.frequency_min) or np.any(frequency > fiber.frequency_max):
        raise ValueError("probe frequencies must be positive and inside the fiber band")
    for section in fiber.sections:
        if section.attenuation < 0. or section.rayleigh_coefficient < 0.:
            raise ValueError("background loss and Rayleigh rates must be nonnegative")
        guide = section.optics
        if guide.frequency_ref <= 0. or guide.effective_index <= 0. or guide.group_index <= 0.:
            raise ValueError("guide reference frequency and indices must be positive")
        if guide.bend_loss is not None and np.any(guide.bend_loss.attenuation < 0.):
            raise ValueError("bend background loss must be nonnegative")

    nodes = []
    for point in fiber.elements:
        block = TwoPort(point.r_left, point.r_right, point.t_left_to_right, point.t_right_to_left)
        block.validate_passivity()
        nodes.append((point.position, 0, block))
    sample = fiber.rayleigh
    if sample is not None:
        with np.errstate(over="ignore", under="ignore"):
            power = np.abs(sample.coefficients)**2
        if np.any(power > 1.):
            raise ValueError("Rayleigh coefficients must satisfy abs(r)**2 <= 1")
        transmission = np.sqrt(1.-power)
        for index, position in enumerate(sample.positions):
            nodes.append((float(position), 1, index))
    nodes.sort(key=lambda item: (item[0], item[1]))

    def compose(left, right, position):
        try:
            return star(left, right, denominator_tolerance=denominator_tolerance)
        except NumericalResonanceError as error:
            raise NumericalResonanceError(error.index, error.denominator, error.tolerance,
                frequency=float(frequency[error.index]), position=position) from error

    zero = np.zeros_like(frequency, dtype=np.complex128)
    one = np.ones_like(zero)
    result = TwoPort(zero, zero, one, one)
    previous = 0.
    blocked = False
    with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
        for position, kind, value in nodes:
            if position > previous:
                p = path.transmission(previous, position, frequency, xp=xp)
                propagation = TwoPort(0., 0., p, p)
                propagation.validate_passivity()
                result = compose(result, propagation, position)
            if kind == 0:
                block = value
            else:
                r = sample.coefficients[value]
                t = transmission[value]
                block = TwoPort(r, -np.conj(r), t, t)
            result = compose(result, block, position)
            previous = position
            if np.all(result.t_left_to_right == 0.):
                blocked = True
                break
        if not blocked:
            p = path.transmission(previous, fiber.length, frequency, xp=xp)
            propagation = TwoPort(0., 0., p, p)
            propagation.validate_passivity()
            result = compose(result, propagation, fiber.length)
    return Response(probe, reflection=result.r_left, transmission=result.t_left_to_right)

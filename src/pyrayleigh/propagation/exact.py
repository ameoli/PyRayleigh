"""Scalar static NumPy reference, using point, Rayleigh and Bragg two-ports."""

import array_api_compat as apc
import numpy as np

from ..backends import complex_arrays
from ..fiber import Fiber, MultiCoreFiber
from ..probe import Probe
from ..scattering import PointReflector
from ..state import FiberState
from ..state.state import FIELD_NAMES
from .cascade import NumericalResonanceError, TwoPort, _TOLERANCE, star
from .phase import optical_path


def _static_blocks(fiber, probe, state, loss_convention):
    """Validate the whole model, then visit propagation and local two-ports."""
    if not isinstance(loss_convention, str) or loss_convention != "background":
        raise ValueError("loss_convention must be 'background'")
    if isinstance(fiber, MultiCoreFiber):
        raise NotImplementedError("static two-port responses currently support one scalar core")
    if not isinstance(fiber, Fiber) or not isinstance(probe, Probe):
        raise TypeError("a Fiber and a Probe are required")
    state =  FiberState() if state is None else state
    if not isinstance(state, FiberState):
        raise TypeError("state must be a FiberState")
    if not state.is_static:
        raise NotImplementedError("two-port responses require a static or frozen state")
    if any(not isinstance(point, PointReflector) for point in fiber.elements):
        raise NotImplementedError("two-port responses currently support PointReflector elements only")
    if fiber.gratings and (fiber.index_perturbation is not None
            or any(not getattr(state, name).is_zero for name in FIELD_NAMES)):
        raise NotImplementedError('Bragg gratings currently require an empty reference state and no index profile')

    path = optical_path(fiber, state)
    parameters = path._parameters(probe.frequency, extra=path.loss._parameters())
    if probe.time is not None:
        parameters.append(probe.time)
    for section in fiber.sections:
        parameters.extend((section.rayleigh_coefficient, section.optics.thermal_expansion))
    for point in fiber.elements:
        parameters.extend((point.r_left, point.r_right, point.t_left_to_right, point.t_right_to_left))
    for grating in fiber.gratings:
        parameters.extend(grating._parameters())
    if fiber.rayleigh is not None:
        parameters.append(fiber.rayleigh.coefficients)
    xp, arrays = complex_arrays(*parameters)
    if not apc.is_numpy_namespace(xp):
        raise NotImplementedError("static two-port responses currently support NumPy only")
    if any(not np.all(np.isfinite(value)) for value in arrays):
        raise ValueError("response parameters must be finite")
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
        nodes.append((point.position, point.position, 0, block))
    sample = fiber.rayleigh
    if sample is not None:
        with np.errstate(over="ignore", under="ignore"):
            power = np.abs(sample.coefficients)**2
        if np.any(power > 1.):
            raise ValueError("Rayleigh coefficients must satisfy abs(r)**2 <= 1")
        transmission = np.sqrt(1.-power)
        for index, position in enumerate(sample.positions):
            nodes.append((float(position), float(position), 1, index))
    for grating in fiber.gratings:
        section = next((item for item in fiber.sections
            if item.start <= grating.start and grating.end <= item.end), None)
        if section is None:
            raise NotImplementedError('a Bragg interval must lie within one section')
        if section.attenuation != 0. or section.optics.bend_loss is not None:
            raise NotImplementedError('background loss inside Bragg intervals is not supported yet')
        if any(grating.start < point.position < grating.end for point in fiber.elements):
            raise NotImplementedError('point elements inside Bragg intervals are not supported yet')
        if sample is not None and np.any((sample.positions > grating.start) & (sample.positions < grating.end)):
            raise NotImplementedError('Rayleigh scatterers inside Bragg intervals are not supported yet')
        block = grating.two_port(frequency, section.optics)
        nodes.append((grating.start, grating.end, 2, block))
    nodes.sort(key=lambda item: (item[0], item[2]))

    def blocks():
        previous = 0.
        for position, end, kind, value in nodes:
            if position > previous:
                p = path.transmission(previous, position, frequency, xp=xp)
                propagation = TwoPort(0., 0., p, p)
                propagation.validate_passivity()
                yield position, propagation
            if kind == 1:
                r = sample.coefficients[value]
                t = transmission[value]
                block = TwoPort(r, -np.conj(r), t, t)
            else:
                block = value
            yield position, block
            previous = end
        p = path.transmission(previous, fiber.length, frequency, xp=xp)
        propagation = TwoPort(0., 0., p, p)
        propagation.validate_passivity()
        yield fiber.length, propagation

    return blocks()


def exact_response(fiber, probe, state=None, *, loss_convention,
                   denominator_tolerance=_TOLERANCE):
    """Left reflection and transmission, with all orders of discrete scattering.

    Section attenuation must be declared as background loss. Only stored
    Rayleigh samples are used; no random draws or grid changes are made.
    A single Fiber, static state and NumPy parameters are supported.
    Uniform Bragg intervals require an unperturbed fiber and one lossless
    section each. Point and Rayleigh scatterers may lie at their boundaries.
    """
    from ..response import Response

    blocks = _static_blocks(fiber, probe, state, loss_convention)
    frequency = np.asarray(probe.frequency)
    zero = np.zeros_like(frequency, dtype=np.complex128)
    one = np.ones_like(zero)
    result = TwoPort(zero, zero, one, one)
    with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
        for position, block in blocks:
            try:
                result = star(result, block, denominator_tolerance=denominator_tolerance)
            except NumericalResonanceError as error:
                raise NumericalResonanceError(error.index, error.denominator, error.tolerance,
                    frequency=float(frequency[error.index]), position=position) from error
            if np.all(result.t_left_to_right == 0.):
                break
    return Response(probe, reflection=result.r_left, transmission=result.t_left_to_right)

"""Scalar static NumPy reference, using point, Rayleigh and Bragg two-ports."""

from dataclasses import replace
import math

import array_api_compat as apc
import numpy as np

from ..backends import complex_arrays
from ..constants import SPEED_OF_LIGHT
from ..fiber import Fiber, MultiCoreFiber
from ..probe import Probe
from ..scattering import PointReflector
from ..state import FiberState, Field
from .cascade import NumericalResonanceError, TwoPort, _TOLERANCE, _check_tolerance, star
from .phase import optical_path


def _uniform_value(field, start, end):
    """Check two interior points per linear piece, without averaging a gradient."""
    if isinstance(field, Field) and field.is_zero:
        return 0.
    profiles = [term.profile for term in field.terms] if isinstance(field, Field) else [field]
    knots = sorted({start, end, *(float(knot) for profile in profiles
        for knot in profile.positions if start < knot < end)})
    positions = []
    for left, right in zip(knots[:-1], knots[1:]):
        a, b = left+.25*(right-left), left+.75*(right-left)
        if not left < a < b < right:
            raise ValueError('Bragg profile interval is too small to sample reliably')
        positions.extend((a, b))
    values = field.value(np.asarray(positions))
    if not np.all(values == values[0]):
        raise NotImplementedError('Bragg perturbations must be uniform over each interval')
    return values[0]


def _bragg_block(grating, frequency, optics, state, denominator_tolerance):
    fields = (state.axial_strain, state.temperature_change)
    edges = sorted({grating.start, grating.end, *(float(knot)
        for field in fields for term in field.terms for knot in term.profile.positions
        if grating.start < knot < grating.end)})
    if grating.segment_length is None:
        loads = [tuple(_uniform_value(field, a, b) for field in fields)
            for a, b in zip(edges[:-1], edges[1:])]
    else:
        refined = []
        for left, right in zip(edges[:-1], edges[1:]):
            count = max(1, math.ceil((right-left)/grating.segment_length))
            refined.extend(np.linspace(left, right, count+1)[:-1])
        edges = np.asarray([*refined, edges[-1]])
        middle = edges[:-1]+.5*np.diff(edges)
        if np.any(middle <= edges[:-1]) or np.any(middle >= edges[1:]):
            raise ValueError('Bragg segments have no representable interior midpoint')
        loads = list(zip(*(field.value(middle) for field in fields)))
        merged_edges, merged_loads = [edges[0]], []
        for right, load in zip(edges[1:], loads):
            if merged_loads and load == merged_loads[-1]:
                merged_edges[-1] = right
            else:
                merged_edges.append(right)
                merged_loads.append(load)
        edges, loads = merged_edges, merged_loads
    if all(load == loads[0] for load in loads):
        return grating.two_port(frequency, optics,
            axial_strain=loads[0][0], temperature_change=loads[0][1])
    result = None
    for left, right, (strain, temperature) in zip(edges[:-1], edges[1:], loads):
        coupling = grating.coupling
        offset = left-grating.start
        if offset != 0.:
            bragg_frequency = SPEED_OF_LIGHT/grating.bragg_wavelength
            # Each piece uses its own left plane, on the same material grating.
            phase = (np.exp(-2j*optics.reference_propagation_constant*offset)
                *np.exp(-2j*optics.propagation_constant_offset(bragg_frequency)*offset))
            coupling = coupling*phase
        piece = replace(grating, start=left, length=right-left, coupling=coupling)
        block = piece.two_port(frequency, optics, axial_strain=strain, temperature_change=temperature)
        try:
            result = block if result is None else star(result, block,
                denominator_tolerance=denominator_tolerance)
        except NumericalResonanceError as error:
            raise NumericalResonanceError(error.index, error.denominator, error.tolerance,
                frequency=float(frequency[error.index]), position=left) from error
    # Each piece is checked separately; allow accumulated round-off here.
    result.validate_passivity(tolerance=_TOLERANCE*len(loads))
    return result


def _static_blocks(fiber, probe, state, loss_convention, denominator_tolerance=_TOLERANCE):
    """Validate the whole model, then visit propagation and local two-ports."""
    _check_tolerance(denominator_tolerance)
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
        for name in ('curvature_x', 'curvature_y'):
            if _uniform_value(getattr(state, name), grating.start, grating.end) != 0.:
                raise NotImplementedError('curvature inside Bragg intervals is not supported yet')
        if (fiber.index_perturbation is not None
                and _uniform_value(fiber.index_perturbation, grating.start, grating.end) != 0.):
            raise NotImplementedError('static index perturbations inside Bragg intervals are not supported yet')
        block = _bragg_block(grating, frequency, section.optics, state, denominator_tolerance)
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
    Each Bragg interval needs one lossless section. Strain and temperature
    may be piecewise constant, split at their profile knots. An explicit
    segment_length enables midpoint approximation of continuous gradients;
    compare refined meshes. Curvature and static index changes stay outside.
    Point and Rayleigh scatterers may lie at the interval boundaries.
    """
    from ..response import Response

    blocks = _static_blocks(fiber, probe, state, loss_convention, denominator_tolerance)
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

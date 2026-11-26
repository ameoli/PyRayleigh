"""Scalar static cascades of point, Rayleigh and Bragg two-ports."""

from dataclasses import replace
import math

import array_api_compat as apc
import numpy as np

from ..backends import complex_arrays, real_arrays
from ..constants import SPEED_OF_LIGHT
from ..fiber import Fiber, MultiCoreFiber
from ..probe import Probe
from ..scattering import PointReflector
from ..state import FiberState, Field, FieldTerm
from .cascade import NumericalResonanceError, TwoPort, _TOLERANCE, _check_tolerance, star
from .loss import _curvature_crossings
from .phase import optical_path


def _midpoint_load(field, start, end):
    """Local linear interpolation when an absolute midpoint cannot be represented."""
    profiles = [term.profile for term in field.terms] if isinstance(field, Field) else [field]
    value, change = 0., 0.
    for profile in profiles:
        for index, (left, right) in enumerate(zip(profile.positions[:-1], profile.positions[1:])):
            if left <= start and end <= right and left < right:
                delta = profile.values[index+1]-profile.values[index]
                fraction = ((start-left)+.5*(end-start))/(right-left)
                value = value+(profile.values[index]+fraction*delta)
                change = change+delta*((end-start)/(right-left))
                break
    return value, change


def _uniform_value(field, start, end):
    """Check two interior points per linear piece, without averaging a gradient."""
    if isinstance(field, Field) and field.is_zero:
        return 0.
    profiles = [term.profile for term in field.terms] if isinstance(field, Field) else [field]
    if all(profile.positions[0] <= start and end <= profile.positions[-1]
            and np.all(profile.values == profile.values[0]) for profile in profiles):
        return sum(profile.values[0] for profile in profiles)
    knots = sorted({start, end, *(float(knot) for profile in profiles
        for knot in profile.positions if start < knot < end)})
    positions, local_values = [], []
    for left, right in zip(knots[:-1], knots[1:]):
        a, b = left+.25*(right-left), left+.75*(right-left)
        if not left < a < b < right:
            value, change = _midpoint_load(field, left, right)
            if change != 0.:
                raise NotImplementedError('Bragg perturbations must be uniform over each interval')
            local_values.append(value)
        else:
            positions.extend((a, b))
    values = np.concatenate((field.value(np.asarray(positions)), np.asarray(local_values)))
    if not np.all(values == values[0]):
        raise NotImplementedError('Bragg perturbations must be uniform over each interval')
    return values[0]


def _bragg_block(grating, frequency, optics, state, denominator_tolerance, *, scatterers=(),
                 attenuation=0., index_perturbation=None):
    """A whole grating, with ordered local scatterers inside."""
    index = Field() if index_perturbation is None else Field((FieldTerm(index_perturbation),))
    table = optics.bend_loss
    bending = () if table is None else (state.curvature_x, state.curvature_y)
    profiles = (grating.apodization, grating.coupling_phase)
    modulation = tuple(Field((FieldTerm(profile),)) for profile in profiles if profile is not None)
    if grating.apodization is not None and np.any(grating.apodization.values < 0.):
        raise ValueError('apodization must be nonnegative')
    fields = (state.axial_strain, state.temperature_change, index)+bending+modulation
    positions = {position for position, _ in scatterers}
    edges = sorted({grating.start, grating.end, *positions, *(float(knot)
        for field in fields for term in field.terms for knot in term.profile.positions
        if grating.start < knot < grating.end)})
    if grating.segment_length is None:
        loads = [tuple(_uniform_value(field, a, b) for field in fields)
            for a, b in zip(edges[:-1], edges[1:])]
    else:
        if bending:
            cuts = set(edges)
            for left, right in zip(edges[:-1], edges[1:]):
                (x, dx), (y, dy) = (_midpoint_load(field, left, right) for field in bending)
                x, y = x-.5*dx, y-.5*dy
                if max(np.hypot(x, y), np.hypot(x+dx, y+dy)) > table.curvatures[-1]:
                    raise ValueError('curvature inside a Bragg interval exceeds the bend-loss table')
                fractions = _curvature_crossings(x, y, dx, dy, table.curvatures, np)
                for fraction in fractions:
                    position = left+float(fraction)*(right-left)
                    if left < position < right:
                        cuts.add(position)
            edges = sorted(cuts)
        refined = []
        for left, right in zip(edges[:-1], edges[1:]):
            count = max(1, math.ceil((right-left)/grating.segment_length))
            refined.extend(np.linspace(left, right, count+1)[:-1])
        edges = np.asarray([*refined, edges[-1]])
        if np.any(np.diff(edges) <= 0.):
            raise ValueError('Bragg segment edges must be distinct')
        middle = edges[:-1]+.5*np.diff(edges)
        loads = list(zip(*(field.value(middle) for field in fields)))
        for index in np.flatnonzero((middle <= edges[:-1]) | (middle >= edges[1:])):
            loads[index] = tuple(_midpoint_load(field, edges[index], edges[index+1])[0] for field in fields)
        merged_edges, merged_loads = [edges[0]], []
        for left, right, load in zip(edges[:-1], edges[1:], loads):
            if merged_loads and load == merged_loads[-1] and left not in positions:
                merged_edges[-1] = right
            else:
                merged_edges.append(right)
                merged_loads.append(load)
        edges, loads = merged_edges, merged_loads
    losses = [attenuation]*len(loads)
    if table is not None:
        losses = [attenuation+table.value(np.hypot(*load[3:5]), frequency) for load in loads]
    if not scatterers and all(load == loads[0] for load in loads):
        return grating.two_port(frequency, optics,
            axial_strain=loads[0][0], temperature_change=loads[0][1],
            attenuation=losses[0], index_change=loads[0][2], denominator_tolerance=denominator_tolerance)
    result = None
    scatterer_index = 0
    for left, right, load, loss in zip(edges[:-1], edges[1:], loads, losses):
        strain, temperature, index_change = load[:3]
        coupling = grating.coupling
        profile_index = 3+len(bending)
        if grating.apodization is not None:
            coupling = coupling*load[profile_index]
            profile_index += 1
        if grating.coupling_phase is not None and load[profile_index] != 0.:
            coupling = coupling*np.exp(1j*load[profile_index])
        offset = left-grating.start
        if offset != 0.:
            bragg_frequency = SPEED_OF_LIGHT/grating.bragg_wavelength
            # Each piece uses its own left plane, on the same material grating.
            phase = (np.exp(-2j*optics.reference_propagation_constant*offset)
                *np.exp(-2j*optics.propagation_constant_offset(bragg_frequency)*offset))
            coupling = coupling*phase
        piece = replace(grating, start=left, length=right-left, coupling=coupling,
            apodization=None, coupling_phase=None)
        block = piece.two_port(frequency, optics, axial_strain=strain,
            temperature_change=temperature, attenuation=loss, index_change=index_change)
        position = left
        try:
            result = block if result is None else star(result, block,
                denominator_tolerance=denominator_tolerance)
            position = right
            while scatterer_index < len(scatterers) and scatterers[scatterer_index][0] == right:
                result = star(result, scatterers[scatterer_index][1],
                    denominator_tolerance=denominator_tolerance)
                scatterer_index += 1
        except NumericalResonanceError as error:
            raise NumericalResonanceError(error.index, error.denominator, error.tolerance,
                frequency=float(frequency[error.index]), position=position) from error
    # Each piece is checked separately; allow accumulated round-off here.
    result.validate_passivity(tolerance=_TOLERANCE*(len(loads)+len(scatterers)))
    return result


def _static_inputs(fiber, probe, state, loss_convention, denominator_tolerance, xp):
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
    xp, arrays = complex_arrays(*parameters, xp=xp)
    return path, xp, arrays


def _static_blocks(fiber, probe, state, loss_convention, denominator_tolerance=_TOLERANCE):
    """NumPy blocks, also used by the direct-path reference."""
    path, xp, arrays = _static_inputs(fiber, probe, state, loss_convention, denominator_tolerance, None)
    if not apc.is_numpy_namespace(xp):
        raise NotImplementedError("the direct-path reference currently supports NumPy only")
    return _numpy_static_blocks(path, probe, xp, arrays, denominator_tolerance)


def _numpy_static_blocks(path, probe, xp, arrays, denominator_tolerance):
    fiber, state = path.fiber, path.state
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
        scatterers = []
        inside = sorted((node for node in nodes if node[2] in (0, 1)
            and grating.start < node[0] < grating.end), key=lambda node: (node[0], node[2]))
        for position, _, kind, value in inside:
            if kind == 1:
                r, t = sample.coefficients[value], transmission[value]
                block = TwoPort(r, -np.conj(r), t, t)
            else:
                block = value
            scatterers.append((float(position), block))
        if inside:
            nodes = [node for node in nodes
                if not (node[2] in (0, 1) and grating.start < node[0] < grating.end)]
        block = _bragg_block(grating, frequency, section.optics, state, denominator_tolerance,
            scatterers=scatterers, attenuation=section.attenuation, index_perturbation=fiber.index_perturbation)
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


def _native_static_blocks(path, probe, xp, arrays, denominator_tolerance):
    fiber = path.fiber
    _, (frequency,) = real_arrays(probe.frequency, xp=xp)
    valid = (xp.isfinite(frequency) & (frequency > 0.)
        & (frequency >= float(fiber.frequency_min)) & (frequency <= float(fiber.frequency_max)))
    for value in arrays[1:]:
        valid = valid & xp.all(xp.isfinite(value))
    for section in fiber.sections:
        guide = section.optics
        _, (alpha, rate, reference, phase_index, group_index) = real_arrays(section.attenuation,
            section.rayleigh_coefficient, guide.frequency_ref, guide.effective_index, guide.group_index, xp=xp)
        valid = valid & (alpha >= 0.) & (rate >= 0.) & (reference > 0.) & (phase_index > 0.) & (group_index > 0.)
        if guide.bend_loss is not None:
            _, (loss,) = real_arrays(guide.bend_loss.attenuation, xp=xp)
            valid = valid & xp.all(loss >= 0.)

    nodes = []
    for point in fiber.elements:
        _, fields = complex_arrays(point.r_left, point.r_right,
            point.t_left_to_right, point.t_right_to_left, xp=xp)
        block = TwoPort(*fields)
        valid = valid & block.passivity_mask()
        nodes.append((float(point.position), float(point.position), 0, block))
    if fiber.rayleigh is not None:
        sample = fiber.rayleigh
        _, (coefficients,) = complex_arrays(sample.coefficients, xp=xp)
        power = xp.real(coefficients*xp.conj(coefficients))
        allowed = xp.isfinite(power) & (power <= 1.)
        valid = valid & xp.all(allowed)
        transmission = xp.sqrt(1.-xp.where(allowed, power, xp.zeros_like(power)))
        for index, position in enumerate(sample.positions):
            r, t = coefficients[index], transmission[index]
            nodes.append((float(position), float(position), 1, TwoPort(r, -xp.conj(r), t, t)))
    from .bragg import _native_bragg_block
    for grating in fiber.gratings:
        section = next((item for item in fiber.sections
            if item.start <= grating.start and grating.end <= item.end), None)
        if section is None:
            raise NotImplementedError('a Bragg interval must lie within one section')
        inside = sorted((node for node in nodes if node[2] in (0, 1)
            and grating.start < node[0] < grating.end), key=lambda node: (node[0], node[2]))
        block, local_valid = _native_bragg_block(grating, frequency, section.optics, path.state,
            denominator_tolerance, attenuation=section.attenuation, index_perturbation=fiber.index_perturbation,
            scatterers=[(position, value) for position, _, _, value in inside], xp=xp)
        valid = valid & local_valid
        nodes = [node for node in nodes if not (node[2] in (0, 1)
            and grating.start < node[0] < grating.end)]
        nodes.append((float(grating.start), float(grating.end), 2, block))
    nodes.sort(key=lambda node: (node[0], node[2]))
    blocks, previous = [], 0.
    for position, end, _, block in nodes:
        if position > previous:
            p = path.transmission(previous, position, frequency, xp=xp)
            propagation = TwoPort(0., 0., p, p)
            valid = valid & propagation.passivity_mask()
            blocks.append((position, propagation))
        blocks.append((position, block))
        previous = end
    p = path.transmission(previous, float(fiber.length), frequency, xp=xp)
    propagation = TwoPort(0., 0., p, p)
    valid = valid & propagation.passivity_mask()
    blocks.append((float(fiber.length), propagation))
    return blocks, valid


def exact_response(fiber, probe, state=None, *, loss_convention,
                   denominator_tolerance=_TOLERANCE, xp=None):
    """Left reflection and transmission, with all orders of discrete scattering.

    Section attenuation must be declared as background loss. Only stored
    Rayleigh samples are used; no random draws or grid changes are made.
    A single Fiber and static or frozen state are required. Point, Rayleigh
    and Bragg scattering keep native coefficients and path loads.
    Invalid native samples return NaN fields, including at the denominator
    cutoff. NumPy keeps its numerical exceptions. Rayleigh transmission
    derivatives require abs(r) < 1; geometry and disorder are not resampled.
    Each Bragg interval needs one section, with its background attenuation.
    Strain, temperature and static index changes are split at profile knots.
    Each piece must be uniform unless segment_length is set.
    An explicit segment_length enables midpoint approximation of continuous
    gradients; compare refined meshes. Curvature only contributes through
    optics.bend_loss. Without a mesh its components must be piecewise constant.
    With segment_length, curvature also uses midpoint samples, splitting at
    table crossings and norm minima. Its upper table limit must hold over
    each linear piece, including its end limits.
    PointReflector and stored Rayleigh scatterers are composed at their
    material positions, including inside gratings.
    Native bend meshes keep a fixed interval count while crossing positions
    remain native. They can be denser than the NumPy mesh at the same
    segment_length; compare refined responses, not equal mesh labels.
    """
    from ..response import Response

    result, _ = _exact_cascade(fiber, probe, state, loss_convention=loss_convention,
        denominator_tolerance=denominator_tolerance, xp=xp, left_only=True)
    return Response(probe, reflection=result.r_left, transmission=result.t_left_to_right)


def exact_two_port(fiber, probe, state=None, *, loss_convention,
                   denominator_tolerance=_TOLERANCE, xp=None, return_validity=False):
    """Full fiber TwoPort at the probe samples, with both external planes.

    The matrix is [[r_left, t_right_to_left], [t_left_to_right, r_right]],
    with shape probe.shape + (2, 2). Coefficients keep their native namespace.
    Static/frozen scalar fibers, background loss and Bragg meshes follow
    exact_response. The original fiber, probe and disorder are not changed.
    return_validity=True also returns a native sample mask for valid physical
    inputs and finite arithmetic above the denominator cutoff, not an error
    estimate. NumPy still raises; invalid native samples have four NaN fields.
    The whole cascade is visited even when left transmission vanishes.
    A singular closed cavity is refused even behind an opaque component;
    exact_response can still give its unaffected left response in that case.
    With JAX, extract the fields or scattering_matrix() inside a jit wrapper.
    """
    if not isinstance(return_validity, (bool, np.bool_)):
        raise TypeError('return_validity must be a fixed bool')
    result, valid = _exact_cascade(fiber, probe, state, loss_convention=loss_convention,
        denominator_tolerance=denominator_tolerance, xp=xp, left_only=False)
    return (result, valid) if return_validity else result


def _exact_cascade(fiber, probe, state, *, loss_convention, denominator_tolerance, xp, left_only):
    path, xp, arrays = _static_inputs(fiber, probe, state, loss_convention, denominator_tolerance, xp)
    is_numpy = apc.is_numpy_namespace(xp)
    if is_numpy:
        blocks = _numpy_static_blocks(path, probe, xp, arrays, denominator_tolerance)
    else:
        with np.errstate(over='ignore', invalid='ignore', divide='ignore', under='ignore'):
            blocks, valid = _native_static_blocks(path, probe, xp, arrays, denominator_tolerance)
    _, (frequency,) = real_arrays(probe.frequency, xp=xp)
    zero = xp.zeros_like(frequency, dtype=xp.complex128)
    one = xp.ones_like(zero)
    result = TwoPort(zero, zero, one, one)
    identity = (zero, zero, one, one)
    policy = 'raise' if is_numpy else 'ignore'
    with np.errstate(over=policy, invalid=policy, divide=policy, under='ignore'):
        for position, block in blocks:
            if not is_numpy and left_only:
                denominator = 1.-result.r_right*block.r_left
                # A closed cavity behind an opaque prefix cannot change the left response.
                hidden = ((result.t_left_to_right == 0.)
                    & (xp.abs(denominator) <= float(denominator_tolerance)))
                active = ~hidden
                old = (result.r_left, result.r_right, result.t_left_to_right, result.t_right_to_left)
                left = TwoPort(*(xp.where(active, value, initial) for value, initial in zip(old, identity)))
                updated = star(left, block, denominator_tolerance=denominator_tolerance)
                fields = (updated.r_left, updated.r_right, updated.t_left_to_right, updated.t_right_to_left)
                result = TwoPort(*(xp.where(active, value, previous) for value, previous in zip(fields, old)))
                continue
            try:
                result = star(result, block, denominator_tolerance=denominator_tolerance)
            except NumericalResonanceError as error:
                raise NumericalResonanceError(error.index, error.denominator, error.tolerance,
                    frequency=float(frequency[error.index]), position=position) from error
            if is_numpy and left_only and np.all(result.t_left_to_right == 0.):
                break
    if is_numpy:
        valid = xp.ones_like(result.r_left, dtype=xp.bool)
    else:
        fields = (result.r_left, result.r_right, result.t_left_to_right, result.t_right_to_left)
        for value in fields:
            valid = valid & xp.isfinite(value)
        result = TwoPort(*(xp.where(valid, value, xp.full_like(value, float('nan'))) for value in fields))
    return result, valid

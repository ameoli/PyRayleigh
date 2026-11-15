"""Reflection and direct transmission at the samples of a probe."""

from dataclasses import dataclass, field, replace

import numpy as np

from .backends import complex_arrays, real_arrays
from .fiber import MultiCoreFiber
from .probe import Probe
from .propagation import multiple_point_response, optical_path, point_response, rayleigh_response
from .propagation.paths import _check_max_order
from .propagation.phase import _GUIDE_PARAMETERS
from .scattering import PointReflector, crosstalk_from_phases
from .state import FiberState
from .state.state import FIELD_NAMES


@dataclass(frozen=True, eq=False)
class Response:
    """Dimensionless field amplitudes for unit light launched at the left port.

    Both arrays have the probe's shape. They are copied as complex128 in
    their native namespace; the original probe is retained as metadata.
    """

    reflection: object = field(repr=False, kw_only=True)
    transmission: object = field(repr=False, kw_only=True)
    probe: Probe

    def __post_init__(self):
        if not isinstance(self.probe, Probe):
            raise TypeError("probe must be a Probe")
        xp, (reflection, transmission) = complex_arrays(self.reflection, self.transmission)
        if reflection.shape != self.probe.shape or transmission.shape != self.probe.shape:
            raise ValueError("response arrays must have the probe's shape")
        object.__setattr__(self, "reflection", xp.asarray(reflection, copy=True))
        object.__setattr__(self, "transmission", xp.asarray(transmission, copy=True))

    @property
    def shape(self):
        return self.probe.shape

    def __repr__(self):
        return f"Response(shape={self.shape})"


@dataclass(frozen=True, eq=False)
class MultiCoreResponse:
    """Native complex128 fields with shape (n_cores,) + probe.shape.

    Core order follows MultiCoreFiber.cores. Both arrays are copied;
    the original shared probe is retained as metadata.
    """

    reflection: object = field(repr=False, kw_only=True)
    transmission: object = field(repr=False, kw_only=True)
    probe: Probe

    def __post_init__(self):
        if not isinstance(self.probe, Probe):
            raise TypeError("probe must be a Probe")
        xp, (reflection, transmission) = complex_arrays(self.reflection, self.transmission)
        if (reflection.ndim != len(self.probe.shape) + 1 or reflection.shape[0] == 0
                or reflection.shape[1:] != self.probe.shape or transmission.shape != reflection.shape):
            raise ValueError("fields must have shape (n_cores,) + probe.shape with n_cores > 0")
        object.__setattr__(self, "reflection", xp.asarray(reflection, copy=True))
        object.__setattr__(self, "transmission", xp.asarray(transmission, copy=True))

    @property
    def shape(self):
        return self.reflection.shape

    @property
    def n_cores(self):
        return self.reflection.shape[0]

    def __repr__(self):
        return f"MultiCoreResponse(shape={self.shape})"


def compute_response(fiber, probe, state=None, *, max_order=1, xp=None):
    """Complex reflection and direct transmission at the probe samples.

    max_order is a fixed integer: 1 gives the direct point, FBG and Rayleigh
    returns; 2 and 3 add repeated PointReflector paths with up to
    2*max_order-1 reflections. Transmission is the direct one-way field.
    A varying state is frozen at each probe.time sample for the whole path,
    including repeated passages. The original probe and disorder are kept.

    MultiCoreFiber returns MultiCoreResponse. Independent cores accept a
    shared state or one FiberState or None per core. Active crosstalk
    supports Rayleigh and direct point returns with a shared state and common
    propagation. Points must share positions and match unique crosstalk grid
    entries; their transmission coefficients must agree between cores.
    Weak FBG returns stay per core, outside crosstalk, as in PyOFDR.
    Multiple point reflections with crosstalk remain unsupported.
    Rayleigh and point coefficients are mixed before propagation. Different fixed path
    layouts raise NotImplementedError; unequal native path parameters give
    NaN fields, keeping the comparison traceable with JAX.
    """
    if not isinstance(probe, Probe):
        raise TypeError("probe must be a Probe")
    _check_max_order(max_order)
    if isinstance(fiber, MultiCoreFiber):
        return _multicore_response(fiber, probe, state, max_order, xp)
    state = FiberState() if state is None else state
    if not isinstance(state, FiberState):
        raise TypeError("state must be a FiberState")
    if not state.is_static and probe.time is None:
        raise ValueError("a time-varying state needs probe.time for every sample")
    path = optical_path(fiber)
    parameters = path.loss._parameters()
    if fiber.rayleigh is not None:
        parameters.append(fiber.rayleigh.coefficients)
    for element in fiber.elements:
        parameters.extend(element._parameters())
        if max_order > 1 and isinstance(element, PointReflector):
            parameters.append(element.r_right)
    for name in FIELD_NAMES:
        state_field = getattr(state, name)
        for term in state_field.terms:
            parameters.append(term.profile.values)
            if term.history is not None:
                parameters.append(term.history.values)
    if not state.is_static:
        parameters.append(probe.time)
    xp, _ = complex_arrays(*path._parameters(probe.frequency, extra=parameters), xp=xp)
    if state.is_static:
        reflection, transmission = _static_response(fiber, probe.frequency, state, max_order, xp)
    else:
        _, (frequency, time) = real_arrays(probe.frequency, probe.time, xp=xp)
        frequency, time = xp.reshape(frequency, (-1,)), xp.reshape(time, (-1,))
        reflected, transmitted = [], []
        for index in range(frequency.shape[0]):
            frozen = state.at_time(time[index], xp=xp)
            r, t = _static_response(fiber, frequency[index], frozen, max_order, xp)
            reflected.append(r)
            transmitted.append(t)
        reflection = (xp.reshape(xp.stack(reflected), probe.shape) if reflected
                      else xp.zeros(probe.shape, dtype=xp.complex128))
        transmission = (xp.reshape(xp.stack(transmitted), probe.shape) if transmitted
                        else xp.zeros(probe.shape, dtype=xp.complex128))
    return Response(probe, reflection=reflection, transmission=transmission)


def _multicore_response(fiber, probe, state, max_order, xp):
    if state is None or isinstance(state, FiberState):
        states = (state,)*fiber.n_cores
    else:
        if isinstance(state, (str, bytes)):
            raise TypeError("state must be a FiberState or a sequence of per-core states")
        try:
            states = tuple(state)
        except TypeError:
            raise TypeError("state must be a FiberState or a sequence of per-core states") from None
        if len(states) != fiber.n_cores:
            raise ValueError("state must contain one entry per core")
        if any(item is not None and not isinstance(item, FiberState) for item in states):
            raise TypeError("per-core states must be FiberState objects or None")
    cores = fiber.cores
    if fiber.crosstalk is not None:
        if max_order > 1 and any(isinstance(element, PointReflector)
                                 for core in cores for element in core.elements):
            raise NotImplementedError("coupled point returns currently require max_order=1")
        if any(item is not states[0] for item in states[1:]):
            raise NotImplementedError("coupled response requires one shared FiberState")
        samples = fiber.crosstalk.mix_rayleigh(tuple(core.rayleigh for core in cores),
                                               fiber.xt_db_per_km, xp=xp)
        cores = tuple(replace(core, rayleigh=sample) for core, sample in zip(cores, samples))
        cores = _coupled_point_cores(fiber, cores, xp)
    responses = tuple(compute_response(core, probe, item, max_order=max_order, xp=xp)
                      for core, item in zip(cores, states))
    xp, _ = complex_arrays(*(item.reflection for item in responses),
                            *(item.transmission for item in responses), xp=xp)
    reflection = xp.stack(tuple(item.reflection for item in responses), axis=0)
    transmission = xp.stack(tuple(item.transmission for item in responses), axis=0)
    if fiber.crosstalk is not None:
        valid = _common_coupled_path(fiber.cores, xp)
        reflection = xp.where(valid, reflection, xp.full_like(reflection, complex(float("nan"))))
        transmission = xp.where(valid, transmission, xp.full_like(transmission, complex(float("nan"))))
    return MultiCoreResponse(probe, reflection=reflection, transmission=transmission)


def _coupled_point_cores(fiber, cores, xp):
    """Mix point left returns on copies, keeping FBGs and transmissions."""
    points = tuple(tuple(element for element in core.elements if isinstance(element, PointReflector))
                   for core in cores)
    reference = points[0]
    positions = tuple(element.position for element in reference)
    if any(tuple(element.position for element in core_points) != positions for core_points in points[1:]):
        raise NotImplementedError("coupled point cores require common reflector positions")
    if not reference:
        return cores
    columns = []
    for position in positions:
        matches = np.flatnonzero(fiber.crosstalk.positions == position)
        if len(matches) != 1:
            raise NotImplementedError("coupled reflectors must match unique crosstalk grid positions")
        columns.append(int(matches[0]))
    count = fiber.n_cores*len(reference)
    xp, arrays = complex_arrays(*(element.r_left for core_points in points for element in core_points),
                                fiber.crosstalk.phases, fiber.xt_db_per_km, xp=xp)
    profile = xp.reshape(xp.stack(arrays[:count]), (fiber.n_cores, len(reference)))
    _, (phases,) = real_arrays(fiber.crosstalk.phases, xp=xp)
    selected = xp.stack(tuple(phases[:, column] for column in columns), axis=1)
    mixed = crosstalk_from_phases(profile, xp.asarray(positions, dtype=xp.float64),
                                  fiber.xt_db_per_km, selected,
                                  topology=fiber.crosstalk.topology, xp=xp)
    return tuple(replace(core, elements=tuple(
        replace(element, r_left=mixed[index, positions.index(element.position)])
        if isinstance(element, PointReflector) else element for element in core.elements))
                 for index, core in enumerate(cores))


def _common_coupled_path(cores, xp):
    """Check fixed layout on the host and compare numerical path data natively."""
    reference = cores[0]
    valid = xp.asarray(True)
    for core in cores[1:]:
        if len(core.sections) != len(reference.sections):
            raise NotImplementedError("coupled cores require a common section layout")
        for left, right in zip(reference.sections, core.sections):
            if (left.start, left.end) != (right.start, right.end):
                raise NotImplementedError("coupled cores require a common section layout")
            values = [(left.attenuation, right.attenuation)]
            values.extend((getattr(left.optics, name), getattr(right.optics, name)) for name in _GUIDE_PARAMETERS)
            first, second = left.optics.bend_loss, right.optics.bend_loss
            if (first is None) != (second is None):
                raise NotImplementedError("coupled cores require common bend-loss grids")
            if first is not None:
                if (not np.array_equal(first.curvatures, second.curvatures)
                        or not np.array_equal(first.frequencies, second.frequencies)):
                    raise NotImplementedError("coupled cores require common bend-loss grids")
                values.append((first.attenuation, second.attenuation))
            for a, b in values:
                _, (a, b) = real_arrays(a, b, xp=xp)
                valid = valid & xp.all(a == b)
        first, second = reference.index_perturbation, core.index_perturbation
        if (first is None) != (second is None):
            raise NotImplementedError("coupled cores require a common index-profile layout")
        if first is not None:
            if not np.array_equal(first.positions, second.positions):
                raise NotImplementedError("coupled cores require a common index-profile layout")
            _, (a, b) = real_arrays(first.values, second.values, xp=xp)
            valid = valid & xp.all(a == b)
        left_points = (element for element in reference.elements if isinstance(element, PointReflector))
        right_points = (element for element in core.elements if isinstance(element, PointReflector))
        for first, second in zip(left_points, right_points):
            _, (a, b, c, d) = complex_arrays(first.t_left_to_right, second.t_left_to_right,
                                             first.t_right_to_left, second.t_right_to_left, xp=xp)
            valid = valid & xp.all(a == b) & xp.all(c == d)
    return valid


def _static_response(fiber, frequency, state, max_order, xp):
    path = optical_path(fiber, state)
    reflection = (point_response(fiber, frequency, state, xp=xp)
                  + rayleigh_response(fiber, frequency, state, xp=xp))
    if max_order > 1:
        reflection = reflection + multiple_point_response(fiber, frequency, state,
                                                           max_order=max_order, xp=xp)
    transmission = path.transmission(0., fiber.length, frequency, xp=xp)
    for element in fiber.elements:
        _, (forward,) = complex_arrays(element.t_left_to_right, xp=xp)
        transmission = transmission*forward
    return reflection, transmission

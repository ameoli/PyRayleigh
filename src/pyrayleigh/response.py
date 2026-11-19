"""Reflection and direct transmission at the samples of a probe."""

from dataclasses import dataclass, field, replace
import math

import array_api_compat as apc
import numpy as np

from .backends import complex_arrays, real_arrays
from .fiber import MultiCoreFiber
from .probe import Probe
from .propagation import multiple_point_response, optical_path, point_response, rayleigh_response
from .propagation.multiple import _multiple_point_returns
from .propagation.paths import _check_max_order, point_reflection_paths
from .propagation.phase import _GUIDE_PARAMETERS
from .scattering import PointReflector, crosstalk_from_phases
from .state import FiberState
from .state.state import FIELD_NAMES


_UNSET = object()


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


def compute_response(fiber, probe, state=None, *, max_order=_UNSET, xp=None,
                     method="truncated", loss_convention=None,
                     denominator_tolerance=_UNSET):
    """Complex reflection and transmission at the probe samples.

    method="truncated" keeps the original model. max_order defaults to 1:
    direct point, FBG and Rayleigh returns; 2 and 3 add repeated PointReflector
    paths with up to 2*max_order-1 reflections. Transmission is direct.
    A varying state is frozen at each probe.time sample for the whole path,
    including repeated passages. The original probe and disorder are kept.

    method="exact" uses the static scalar NumPy two-port cascade, including
    Rayleigh depletion. It requires loss_convention="background" and rejects
    an explicitly supplied max_order. denominator_tolerance is optional and
    forwarded to exact_response. These options are exclusive to exact.
    It does not support time-varying states, multicore fibers or WeakFBG.
    Intervals in fiber.gratings accept piecewise constant strain and temperature,
    splitting at profile knots. Set grating.segment_length [m] to approximate
    continuous gradients with constant midpoint loads, and compare finer meshes.
    Each needs one lossless section, without internal curvature or static index
    changes. PointReflector and stored Rayleigh samples may lie inside gratings,
    using their supplied local coefficients. Outside profiles use the normal
    path model.

    MultiCoreFiber returns MultiCoreResponse. Independent cores accept a
    shared state or one FiberState or None per core. Active crosstalk
    supports Rayleigh and point returns with a shared state and common
    propagation. Points must share positions and match unique crosstalk grid
    entries; their transmission coefficients must agree between cores.
    Weak FBG returns stay per core, outside crosstalk, as in PyOFDR.
    Multiple returns use the original point coefficients before mixing their
    fields. Each apparent position must match one crosstalk grid entry,
    allowing floating-point round-off. No clipping or interpolation is done.
    Rayleigh and direct point coefficients are mixed before propagation.
    Different fixed path layouts raise NotImplementedError; unequal native
    path parameters give NaN fields, keeping the comparison traceable with JAX.
    """
    if not isinstance(probe, Probe):
        raise TypeError("probe must be a Probe")
    if not isinstance(method, str):
        raise TypeError("method must be 'truncated' or 'exact'")
    if method not in ("truncated", "exact"):
        raise ValueError("method must be 'truncated' or 'exact'")
    if method == "exact":
        if max_order is not _UNSET:
            raise ValueError("max_order must be omitted for method='exact'")
        if xp is not None and not apc.is_numpy_namespace(xp):
            raise NotImplementedError("method='exact' currently supports NumPy only")
        from .propagation.exact import exact_response
        options = {}
        if denominator_tolerance is not _UNSET:
            options["denominator_tolerance"] = denominator_tolerance
        return exact_response(fiber, probe, state, loss_convention=loss_convention, **options)
    if loss_convention is not None or denominator_tolerance is not _UNSET:
        raise ValueError("loss_convention and denominator_tolerance require method='exact'")
    max_order = 1 if max_order is _UNSET else max_order
    _check_max_order(max_order)
    if isinstance(fiber, MultiCoreFiber):
        if any(core.gratings for core in fiber.cores):
            raise NotImplementedError("Bragg gratings require the scalar method='exact'")
        return _multicore_response(fiber, probe, state, max_order, xp)
    state = FiberState() if state is None else state
    if not isinstance(state, FiberState):
        raise TypeError("state must be a FiberState")
    if not state.is_static and probe.time is None:
        raise ValueError("a time-varying state needs probe.time for every sample")
    path = optical_path(fiber)
    if fiber.gratings:
        raise NotImplementedError("Bragg gratings require method='exact'")
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
        if any(item is not states[0] for item in states[1:]):
            raise NotImplementedError("coupled response requires one shared FiberState")
        samples = fiber.crosstalk.mix_rayleigh(tuple(core.rayleigh for core in cores),
                                               fiber.xt_db_per_km, xp=xp)
        cores = tuple(replace(core, rayleigh=sample) for core, sample in zip(cores, samples))
        cores = _coupled_point_cores(fiber, cores, xp)
    order = 1 if fiber.crosstalk is not None else max_order
    responses = tuple(compute_response(core, probe, item, max_order=order, xp=xp)
                      for core, item in zip(cores, states))
    xp, _ = complex_arrays(*(item.reflection for item in responses),
                            *(item.transmission for item in responses),
                            *(element.r_right for core in fiber.cores for element in core.elements
                              if max_order > 1 and isinstance(element, PointReflector)), xp=xp)
    reflection = xp.stack(tuple(item.reflection for item in responses), axis=0)
    transmission = xp.stack(tuple(item.transmission for item in responses), axis=0)
    if fiber.crosstalk is not None:
        if max_order > 1:
            reflection = reflection + _coupled_multiple_response(fiber, probe, states[0], max_order, xp)
        valid = _common_coupled_path(fiber.cores, xp)
        reflection = xp.where(valid, reflection, xp.full_like(reflection, complex(float("nan"))))
        transmission = xp.where(valid, transmission, xp.full_like(transmission, complex(float("nan"))))
    return MultiCoreResponse(probe, reflection=reflection, transmission=transmission)


def _coupled_multiple_response(fiber, probe, state, max_order, xp):
    """Mix complete source-core ghost fields using the apparent-position phases."""
    reference = fiber.cores[0]
    routes = tuple(route for route in point_reflection_paths(reference, max_order=max_order) if len(route) > 1)
    positions = tuple(math.fsum(reference.elements[index].position if event % 2 == 0
                               else -reference.elements[index].position
                               for event, index in enumerate(route)) for route in routes)
    columns = []
    for position in positions:
        matches = np.flatnonzero(np.isclose(fiber.crosstalk.positions, position,
                                            rtol=8.*np.finfo(float).eps, atol=0.))
        if len(matches) != 1:
            raise NotImplementedError("ghost positions must match unique crosstalk grid positions")
        columns.append(int(matches[0]))
    state = FiberState() if state is None else state
    _, (phases,) = real_arrays(fiber.crosstalk.phases, xp=xp)
    _, (frequency,) = real_arrays(probe.frequency, xp=xp)

    def static(frequency, frozen):
        shape = (fiber.n_cores,) + frequency.shape
        result = xp.zeros(shape, dtype=xp.complex128)
        returns = tuple(_multiple_point_returns(core, frequency, frozen, max_order=max_order, xp=xp)[2]
                        for core in fiber.cores)
        count = math.prod(frequency.shape)
        for position, column, fields in zip(positions, columns, zip(*returns)):
            profile = xp.reshape(xp.stack(tuple(field for _, field in fields), axis=0), (fiber.n_cores, count))
            selected = xp.broadcast_to(phases[:, column, None], (len(fiber.crosstalk.pairs), count))
            mixed = crosstalk_from_phases(profile, xp.full((count,), position, dtype=xp.float64),
                                          fiber.xt_db_per_km, selected,
                                          topology=fiber.crosstalk.topology, xp=xp)
            result = result + xp.reshape(mixed, shape)
        return result

    if state.is_static:
        return static(frequency, state)
    _, (time,) = real_arrays(probe.time, xp=xp)
    frequency, time = xp.reshape(frequency, (-1,)), xp.reshape(time, (-1,))
    values = [static(frequency[index], state.at_time(time[index], xp=xp)) for index in range(frequency.shape[0])]
    if not values:
        return xp.zeros((fiber.n_cores,) + probe.shape, dtype=xp.complex128)
    return xp.reshape(xp.stack(values, axis=1), (fiber.n_cores,) + probe.shape)


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
        matches = np.flatnonzero(np.isclose(fiber.crosstalk.positions, position,
                                            rtol=8.*np.finfo(float).eps, atol=0.))
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

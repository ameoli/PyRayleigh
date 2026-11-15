"""Reflection and direct transmission at the samples of a probe."""

from dataclasses import dataclass, field

from .backends import complex_arrays, real_arrays
from .fiber import MultiCoreFiber
from .probe import Probe
from .propagation import multiple_point_response, optical_path, point_response, rayleigh_response
from .propagation.paths import _check_max_order
from .scattering import PointReflector
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

    An independent MultiCoreFiber returns MultiCoreResponse. state may be
    shared or contain one FiberState or None per core. Active multicore
    crosstalk is not evaluated by this response function yet.
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
    if fiber.crosstalk is not None:
        raise NotImplementedError("multicore response with active crosstalk is not implemented")
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
    responses = tuple(compute_response(core, probe, item, max_order=max_order, xp=xp)
                      for core, item in zip(fiber.cores, states))
    xp, _ = complex_arrays(*(item.reflection for item in responses),
                            *(item.transmission for item in responses), xp=xp)
    return MultiCoreResponse(probe,
                             reflection=xp.stack(tuple(item.reflection for item in responses), axis=0),
                             transmission=xp.stack(tuple(item.transmission for item in responses), axis=0))


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

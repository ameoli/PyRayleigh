"""Reflection and direct transmission at the samples of a probe."""

from dataclasses import dataclass, field

from .backends import complex_arrays, real_arrays
from .probe import Probe
from .propagation import optical_path, point_response, rayleigh_response
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


def compute_response(fiber, probe, state=None, *, xp=None):
    """One reflection per path and unreflected transmission at probe samples.

    A varying state is frozen at each probe.time sample, throughout that
    sample's propagation. The original probe and stored disorder are kept.
    """
    if not isinstance(probe, Probe):
        raise TypeError("probe must be a Probe")
    state = FiberState() if state is None else state
    if not isinstance(state, FiberState):
        raise TypeError("state must be a FiberState")
    if not state.is_static and probe.time is None:
        raise ValueError("a time-varying state needs probe.time for every sample")
    path = optical_path(fiber)
    parameters = [section.attenuation for section in fiber.sections]
    if fiber.rayleigh is not None:
        parameters.append(fiber.rayleigh.coefficients)
    for element in fiber.elements:
        parameters.extend((element.r_left, element.t_left_to_right, element.t_right_to_left))
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
        reflection, transmission = _static_response(fiber, probe.frequency, state, xp)
    else:
        _, (frequency, time) = real_arrays(probe.frequency, probe.time, xp=xp)
        frequency, time = xp.reshape(frequency, (-1,)), xp.reshape(time, (-1,))
        reflected, transmitted = [], []
        for index in range(frequency.shape[0]):
            frozen = state.at_time(time[index], xp=xp)
            r, t = _static_response(fiber, frequency[index], frozen, xp)
            reflected.append(r)
            transmitted.append(t)
        reflection = (xp.reshape(xp.stack(reflected), probe.shape) if reflected
                      else xp.zeros(probe.shape, dtype=xp.complex128))
        transmission = (xp.reshape(xp.stack(transmitted), probe.shape) if transmitted
                        else xp.zeros(probe.shape, dtype=xp.complex128))
    return Response(probe, reflection=reflection, transmission=transmission)


def _static_response(fiber, frequency, state, xp):
    path = optical_path(fiber, state)
    reflection = (point_response(fiber, frequency, state, xp=xp)
                  + rayleigh_response(fiber, frequency, state, xp=xp))
    transmission = path.transmission(0., fiber.length, frequency, xp=xp)
    for element in fiber.elements:
        _, (forward,) = complex_arrays(element.t_left_to_right, xp=xp)
        transmission = transmission*forward
    return reflection, transmission

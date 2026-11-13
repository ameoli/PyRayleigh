"""Reflection and direct transmission at the samples of a probe."""

from dataclasses import dataclass, field

from .backends import complex_arrays
from .probe import Probe
from .propagation import optical_path, point_response, rayleigh_response


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
    """Static response: one reflection per path and unreflected transmission.

    The reflected amplitudes are summed before taking any power. Forward
    transmission crosses every point element once. Stored Rayleigh samples
    leave this direct transmission unchanged, as in the current loss model.
    """
    if not isinstance(probe, Probe):
        raise TypeError("probe must be a Probe")
    path = optical_path(fiber, state)
    parameters = [section.attenuation for section in fiber.sections]
    if fiber.rayleigh is not None:
        parameters.append(fiber.rayleigh.coefficients)
    for element in fiber.elements:
        parameters.extend((element.r_left, element.t_left_to_right, element.t_right_to_left))
    xp, _ = complex_arrays(*path._parameters(probe.frequency, extra=parameters), xp=xp)
    reflection = (point_response(fiber, probe.frequency, state, xp=xp)
                  + rayleigh_response(fiber, probe.frequency, state, xp=xp))
    transmission = path.transmission(0., fiber.length, probe.frequency, xp=xp)
    for element in fiber.elements:
        _, (forward,) = complex_arrays(element.t_left_to_right, xp=xp)
        transmission = transmission*forward
    return Response(probe, reflection=reflection, transmission=transmission)

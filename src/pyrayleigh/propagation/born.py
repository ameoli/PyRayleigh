"""Direct point-reflector returns, with one reflection per contribution."""

from ..backends import complex_arrays, real_arrays
from .phase import optical_path


def point_response(fiber, frequency, state=None, *, xp=None):
    """Complex reflection at the left port from the discrete point elements.

    Each r_left is weighted by distributed round-trip propagation and by
    t_left_to_right*t_right_to_left of every preceding element. Geometry
    is fixed; evaluation uses the supplied coefficients without mutation.
    """
    path = optical_path(fiber, state)
    parameters = [section.attenuation for section in fiber.sections]
    for element in fiber.elements:
        parameters.extend((element.r_left, element.t_left_to_right, element.t_right_to_left))
    xp, _ = complex_arrays(*path._parameters(frequency, extra=parameters), xp=xp)
    _, (frequency,) = real_arrays(frequency, xp=xp)
    response = xp.zeros_like(frequency, dtype=xp.complex128)
    through = xp.asarray(1., dtype=xp.complex128)
    for element in fiber.elements:
        _, (reflection, forward, reverse) = complex_arrays(
            element.r_left, element.t_left_to_right, element.t_right_to_left, xp=xp)
        propagation = path.round_trip(0., element.position, frequency, xp=xp)
        response = response + reflection*through*propagation
        through = through*forward*reverse
    return response

"""Direct returns from stored Rayleigh disorder and point reflectors."""

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


def rayleigh_response(fiber, frequency, state=None, *, xp=None):
    """Complex Rayleigh reflection at the left port, with one reflection per path.

    Samples retain their supplied local coefficients. Each return crosses
    every point element at or before its position in both directions.
    Rayleigh samples themselves add no transmission step, as in PyOFDR.
    """
    path = optical_path(fiber, state)
    sample = fiber.rayleigh
    parameters = [section.attenuation for section in fiber.sections]
    if sample is not None:
        parameters.append(sample.coefficients)
        for element in fiber.elements:
            parameters.extend((element.t_left_to_right, element.t_right_to_left))
    xp, _ = complex_arrays(*path._parameters(frequency, extra=parameters), xp=xp)
    _, (frequency,) = real_arrays(frequency, xp=xp)
    response = xp.zeros_like(frequency, dtype=xp.complex128)
    if sample is None:
        return response
    _, (coefficients,) = complex_arrays(sample.coefficients, xp=xp)
    through = xp.asarray(1., dtype=xp.complex128)
    element_index = 0
    for index, position in enumerate(sample.positions):
        position = float(position)
        while (element_index < len(fiber.elements)
               and fiber.elements[element_index].position <= position):
            element = fiber.elements[element_index]
            _, (forward, reverse) = complex_arrays(
                element.t_left_to_right, element.t_right_to_left, xp=xp)
            through = through*forward*reverse
            element_index += 1
        propagation = path.round_trip(0., position, frequency, xp=xp)
        response = response + coefficients[index]*through*propagation
    return response

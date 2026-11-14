"""Direct returns from stored Rayleigh disorder and localized elements."""

from ..backends import complex_arrays, real_arrays
from ..scattering import WeakFBG
from .phase import optical_path


def point_response(fiber, frequency, state=None, *, xp=None):
    """Complex reflection at the left port from localized elements.

    Each local reflection is weighted by distributed round-trip propagation and by
    t_left_to_right*t_right_to_left of every preceding element. WeakFBG
    resonances use local total strain and temperature with the section's
    phase sensitivities; their reference bandwidth stays fixed. Geometry
    and stored parameters are evaluated without mutation.
    """
    path = optical_path(fiber, state)
    parameters = path.loss._parameters()
    for element in fiber.elements:
        parameters.extend(element._parameters())
    xp, _ = complex_arrays(*path._parameters(frequency, extra=parameters), xp=xp)
    _, (frequency,) = real_arrays(frequency, xp=xp)
    response = xp.zeros_like(frequency, dtype=xp.complex128)
    through = xp.asarray(1., dtype=xp.complex128)
    for element in fiber.elements:
        if isinstance(element, WeakFBG):
            optics = fiber.sections[-1].optics
            for section in fiber.sections:
                if element.position < section.end:
                    optics = section.optics
                    break
            strain = path.state.axial_strain.value(element.position, xp=xp)
            temperature = path.state.temperature_change.value(element.position, xp=xp)
            shift = optics.relative_phase_change(strain, temperature, xp=xp)
            reflection = element.reflection(frequency, optics.effective_index, shift, xp=xp)
            forward = reverse = xp.asarray(1., dtype=xp.complex128)
        else:
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
    parameters = path.loss._parameters()
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

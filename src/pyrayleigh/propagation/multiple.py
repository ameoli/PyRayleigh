"""Additional returns from repeated reflections between point elements."""

from itertools import combinations

from ..backends import complex_arrays, real_arrays
from ..scattering import PointReflector
from .paths import point_reflection_paths
from .phase import optical_path


def multiple_point_response(fiber, frequency, state=None, *, max_order=2, xp=None):
    """Sum the extra left-port returns of orders 2 through max_order.

    Order N has 2*N-1 reflections; max_order=1 gives zero. The state must
    be static, as for point_response. Each leg uses its actual distributed
    phase and loss, including repeated passages through the same section.
    Points use r_left or r_right at a bounce, and the appropriate directional
    transmission when crossed. Distributed propagation is reciprocal.
    Only PointReflector objects bounce here. Acquisition range is left to
    the caller; the result has frequency.shape and dtype complex128.
    """
    routes = tuple(route for route in point_reflection_paths(fiber, max_order=max_order)
                    if len(route) > 1)
    path = optical_path(fiber, state)
    parameters = path.loss._parameters()
    for element in fiber.elements:
        parameters.extend((element.t_left_to_right, element.t_right_to_left))
        if isinstance(element, PointReflector):
            parameters.extend((element.r_left, element.r_right))
    xp, _ = complex_arrays(*path._parameters(frequency, extra=parameters), xp=xp)
    _, (frequency,) = real_arrays(frequency, xp=xp)
    response = xp.zeros_like(frequency, dtype=xp.complex128)
    if not routes:
        return response

    indices = (-1, *(index for index, element in enumerate(fiber.elements)
                      if isinstance(element, PointReflector)))
    travel = {}
    for left, right in combinations(indices, 2):
        start = 0. if left == -1 else fiber.elements[left].position
        end = fiber.elements[right].position
        forward = reverse = path.transmission(start, end, frequency, xp=xp)
        for index in range(left + 1, right):
            element = fiber.elements[index]
            _, (t_forward, t_reverse) = complex_arrays(
                element.t_left_to_right, element.t_right_to_left, xp=xp)
            forward = forward*t_forward
            reverse = reverse*t_reverse
        travel[left, right] = forward
        travel[right, left] = reverse

    for route in routes:
        amplitude = xp.ones_like(frequency, dtype=xp.complex128)
        previous = -1
        for event, current in enumerate(route):
            element = fiber.elements[current]
            reflection = element.r_left if event % 2 == 0 else element.r_right
            _, (reflection,) = complex_arrays(reflection, xp=xp)
            amplitude = amplitude*travel[previous, current]*reflection
            previous = current
        response = response + amplitude*travel[previous, -1]
    return response

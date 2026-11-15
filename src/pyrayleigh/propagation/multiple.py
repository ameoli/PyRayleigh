"""Additional returns from repeated reflections between point elements."""

from itertools import combinations
import math

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
    xp, frequency, returns = _multiple_point_returns(fiber, frequency, state, max_order=max_order, xp=xp)
    response = xp.zeros_like(frequency, dtype=xp.complex128)
    for _, amplitude in returns:
        response = response + amplitude
    return response


def _multiple_point_returns(fiber, frequency, state=None, *, max_order=2, xp=None):
    """Return the namespace, native frequencies and an iterator of (position, field).

    Position is half the geometric path length, without binning or clipping.
    Fields include all actual propagation legs and original point coefficients.
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
    if not routes:
        return xp, frequency, iter(())

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

    def returns():
        for route in routes:
            amplitude = xp.ones_like(frequency, dtype=xp.complex128)
            previous = -1
            for event, current in enumerate(route):
                element = fiber.elements[current]
                reflection = element.r_left if event % 2 == 0 else element.r_right
                _, (reflection,) = complex_arrays(reflection, xp=xp)
                amplitude = amplitude*travel[previous, current]*reflection
                previous = current
            position = math.fsum(fiber.elements[current].position if event % 2 == 0
                                  else -fiber.elements[current].position
                                  for event, current in enumerate(route))
            yield position, amplitude*travel[previous, -1]
    return xp, frequency, returns()

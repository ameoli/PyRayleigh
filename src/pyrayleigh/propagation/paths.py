"""Fixed geometry of returns between discrete reflectors."""

from itertools import product
from numbers import Integral

from ..fiber import Fiber
from ..scattering import PointReflector


def point_reflection_paths(fiber, *, max_order=2):
    """Return tuples of reflection-event indices into fiber.elements.

    Order N has 2*N-1 events, starting from the left and ending at the
    left port. Orders 1 (direct), 2 and 3 follow PyOFDR's convention.
    Paths are ordered by N, then lexicographically. Only PointReflector
    objects reflect here; Rayleigh and WeakFBG returns are separate.
    Coefficients are not inspected, even when zero: paths are fixed
    geometry, independent of trainable data. No bins or acquisition
    range are imposed; total travel can exceed twice the fiber length.
    """
    if not isinstance(fiber, Fiber):
        raise TypeError("fiber must be a Fiber")
    if isinstance(max_order, bool) or not isinstance(max_order, Integral):
        raise TypeError("max_order must be a fixed integer")
    if not 1 <= max_order <= 3:
        raise ValueError("max_order must be 1, 2 or 3")
    indices = tuple(index for index, element in enumerate(fiber.elements)
                    if isinstance(element, PointReflector))
    paths = []
    for order in range(1, max_order + 1):
        for path in product(indices, repeat=2*order - 1):
            if all(path[event] < path[event - 1] and path[event] < path[event + 1]
                   for event in range(1, len(path), 2)):
                paths.append(path)
    return tuple(paths)

"""Fixed geometry of returns between discrete reflectors."""

from itertools import product
import math
from numbers import Integral, Real

from ..fiber import Fiber
from ..scattering import PointReflector


def _check_max_order(max_order):
    if isinstance(max_order, bool) or not isinstance(max_order, Integral):
        raise TypeError("max_order must be a fixed integer")
    if not 1 <= max_order <= 3:
        raise ValueError("max_order must be 1, 2 or 3")


def reflection_paths_from_positions(positions, *, max_order=2):
    """Reflection-event indices on fixed, strictly increasing positions.

    Positions are finite and nonnegative; integer spatial bins also work.
    Orders 1 through 3 include direct returns. Coefficients are not inspected.
    """
    _check_max_order(max_order)
    positions = tuple(positions)
    if any(isinstance(value, bool) or not isinstance(value, Real) for value in positions):
        raise TypeError("positions must contain fixed real numbers")
    if (any(not math.isfinite(value) or value < 0. for value in positions)
            or any(right <= left for left, right in zip(positions[:-1], positions[1:]))):
        raise ValueError("positions must be finite, nonnegative and strictly increasing")
    paths = []
    for order in range(1, max_order + 1):
        for path in product(range(len(positions)), repeat=2*order - 1):
            if all(positions[path[event]] < positions[path[event - 1]]
                   and positions[path[event]] < positions[path[event + 1]]
                   for event in range(1, len(path), 2)):
                paths.append(path)
    return tuple(paths)


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
    if fiber.gratings:
        raise NotImplementedError('point paths do not include extended Bragg gratings')
    indices = tuple(index for index, element in enumerate(fiber.elements)
                    if isinstance(element, PointReflector))
    positions = [fiber.elements[index].position for index in indices]
    paths = reflection_paths_from_positions(positions, max_order=max_order)
    return tuple(tuple(indices[index] for index in path) for path in paths)

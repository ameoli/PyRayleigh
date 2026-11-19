"""Coupled Rayleigh refinement on fixed material geometry."""

import math
from numbers import Real

import numpy as np

from .backends import complex_arrays, real_arrays
from .fiber import Fiber, MultiCoreFiber
from .propagation import optical_path
from .scattering import RayleighRealization, split_rayleigh_coefficients


def refine_rayleigh_from_normals(fiber, normals, *, reference_frequency, xp=None):
    """Return two left-edge children per stored parent, with supplied normals.

    The phase frame is the unperturbed fiber at a fixed frequency [Hz].
    Parents must have the section rate's Gaussian prior and lie at left edges.
    No cell may cross a section boundary. Geometry is fixed host data; optical
    parameters, coefficients and normals stay native for differentiation.
    """
    if isinstance(fiber, MultiCoreFiber):
        raise NotImplementedError("refine one core at a time")
    if not isinstance(fiber, Fiber):
        raise TypeError("fiber must be a Fiber")
    sample = fiber.rayleigh
    if sample is None:
        raise ValueError("fiber needs a stored Rayleigh realization")
    if (isinstance(reference_frequency, (bool, np.bool_))
            or not isinstance(reference_frequency, Real)):
        raise TypeError("reference_frequency must be a fixed real number")
    if (not math.isfinite(reference_frequency)
            or not fiber.frequency_min <= reference_frequency <= fiber.frequency_max):
        raise ValueError("reference_frequency must be finite and inside the fiber band")
    if not np.array_equal(sample.positions, sample.edges[:-1]):
        raise NotImplementedError("refinement requires left-edge parent positions")

    widths = sample.cell_lengths()
    middle = sample.edges[:-1]+.5*widths
    if np.any(middle <= sample.edges[:-1]) or np.any(middle >= sample.edges[1:]):
        raise ValueError("a parent cell has no representable interior midpoint")
    rates = []
    for start, end in zip(sample.edges[:-1], sample.edges[1:]):
        section = next((item for item in fiber.sections if item.start <= start and end <= item.end), None)
        if section is None:
            raise ValueError("parent cells must not cross section boundaries")
        rates.append(section.rayleigh_coefficient)
    path = optical_path(fiber)
    parameters = path._parameters(reference_frequency, extra=(sample.coefficients, normals, *rates))
    xp, _ = complex_arrays(*parameters, xp=xp)
    _, rates = real_arrays(*rates, xp=xp)
    rate = xp.stack(rates)
    fraction = xp.asarray((middle-sample.edges[:-1])/widths, dtype=xp.float64)
    children = split_rayleigh_coefficients(sample.coefficients, normals, rate,
        xp.asarray(widths, dtype=xp.float64), left_fraction=fraction, xp=xp)
    corrections = []
    for start, midpoint in zip(sample.edges[:-1], middle):
        parent_phase = path.phase_factor(0., float(start), reference_frequency, round_trip=True, xp=xp)
        child_phase = path.phase_factor(0., float(midpoint), reference_frequency, round_trip=True, xp=xp)
        corrections.append(parent_phase/child_phase)
    coefficients = xp.stack((children[:, 0], children[:, 1]*xp.stack(corrections)), axis=-1)
    edges = np.empty(2*sample.cell_count+1)
    edges[::2], edges[1::2] = sample.edges, middle
    return RayleighRealization(xp.reshape(coefficients, (-1,)), edges[:-1], edges)

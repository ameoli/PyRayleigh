"""Distributed power loss in the material coordinate."""

from dataclasses import dataclass, field

import numpy as np

from ..backends import real_arrays
from ..fiber import Fiber
from ..state import FiberState

_GAUSS_POINTS, _GAUSS_WEIGHTS = np.polynomial.legendre.leggauss(16)


def _curvature_crossings(x, y, dx, dy, levels, xp):
    """Fixed-size native split list, including the minimum of |kappa|."""
    a = dx*dx + dy*dy
    b = 2.*(x*dx + y*dy)
    safe_a = xp.where(a > 0., a, xp.ones_like(a))
    minimum = xp.where(a > 0., xp.clip(-b/(2.*safe_a), 0., 1.), xp.zeros_like(a))
    roots = [xp.zeros_like(a), xp.ones_like(a), minimum]
    for level in levels:
        c = x*x + y*y - float(level)**2
        discriminant = b*b - 4.*a*c
        valid = (a > 0.) & (discriminant > 0.)
        square_root = xp.sqrt(xp.where(valid, discriminant, xp.ones_like(discriminant)))
        q = -.5*(b + xp.where(b >= 0., square_root, -square_root))
        q = xp.where(valid, q, xp.ones_like(q))
        for root in (q/safe_a, c/q):
            roots.append(xp.where(valid, xp.clip(root, 0., 1.), xp.zeros_like(root)))
    return xp.sort(xp.stack(roots))


def _bend_piece(low, high, table, state, position, frequency, xp):
    inside = xp.asarray([low + .25*(high - low), low + .75*(high - low)], dtype=xp.float64)
    ends = []
    for component in (state.curvature_x, state.curvature_y):
        values = component.value(inside, xp=xp)
        change = 2.*(values[1] - values[0])
        ends.append((values[0] - .25*change, change))
    (x, dx), (y, dy) = ends
    boundaries = _curvature_crossings(x, y, dx, dy, table.curvatures, xp)
    covered = xp.clip((position - low)/(high - low), 0., 1.)
    left = xp.minimum(covered[..., None], boundaries[:-1])
    right = xp.minimum(covered[..., None], boundaries[1:])
    half = .5*(right - left)
    nodes = xp.asarray(_GAUSS_POINTS.tolist(), dtype=xp.float64)
    weights = xp.asarray(_GAUSS_WEIGHTS.tolist(), dtype=xp.float64)
    fractions = left[..., None] + half[..., None]*(1. + nodes)
    squared = (x + fractions*dx)**2 + (y + fractions*dy)**2
    active = (half[..., None] > 0.) & (squared > 0.)
    magnitude = xp.sqrt(xp.where(active, squared, xp.ones_like(squared)))
    magnitude = xp.where(active, magnitude, xp.zeros_like(magnitude))
    rates = table.value(magnitude, frequency, xp=xp)
    scale = (high - low)*half[..., None]*weights
    scale = xp.reshape(scale, scale.shape + (1,)*frequency.ndim)
    return xp.sum(rates*scale, axis=(position.ndim, position.ndim + 1))


@dataclass(frozen=True)
class FiberLoss:
    """Full section attenuation, with the same convention as PyOFDR.

    A stored Rayleigh realization leaves the distributed coefficient
    unchanged. Loss is integrated over material lengths. Bend loss uses
    a static curvature state and each section's optional table.
    """

    fiber: Fiber
    state: FiberState = field(default_factory=FiberState)

    def __post_init__(self):
        if not isinstance(self.fiber, Fiber):
            raise TypeError("fiber must be a Fiber")
        if not isinstance(self.state, FiberState):
            raise TypeError("state must be a FiberState")
        if not self.state.is_static:
            raise NotImplementedError("FiberLoss requires a static FiberState")

    def _parameters(self):
        parameters = [section.attenuation for section in self.fiber.sections]
        for section in self.fiber.sections:
            if section.optics.bend_loss is not None:
                parameters.append(section.optics.bend_loss.attenuation)
        for component in (self.state.curvature_x, self.state.curvature_y):
            parameters.extend(term.profile.values for term in component.terms)
        return parameters

    def attenuation_integral(self, position, frequency, *, xp=None):
        """A(s, nu) = int_0^s alpha ds, shape position.shape + frequency.shape.

        Finite positions [m] are clipped to the fiber extent. Frequencies
        are absolute Hz. Rates must be finite and nonnegative [1/m].

        Bend integration splits at profile knots, table crossings and norm
        minima. A fixed 16-point Gauss rule is exact for constant curvature;
        linear curvature is numerical, without an adaptive error estimate.
        Derivatives exclude table transitions and tangential crossings.
        """
        xp, arrays = real_arrays(position, frequency, *self._parameters(), xp=xp)
        position, frequency = arrays[:2]
        position_shape, frequency_shape = position.shape, frequency.shape
        loss = xp.zeros(position_shape + frequency_shape, dtype=xp.float64)
        knots = [float(knot) for component in (self.state.curvature_x, self.state.curvature_y)
                 for term in component.terms for knot in term.profile.positions]
        for section in self.fiber.sections:
            covered = xp.clip(position - float(section.start), 0., float(section.length))
            covered = xp.reshape(covered, position_shape + (1,)*len(frequency_shape))
            loss = loss + covered*section.attenuation_at(frequency, xp=xp)
            table = section.optics.bend_loss
            if table is None or not knots:
                continue
            edges = sorted({float(section.start), float(section.end),
                            *(knot for knot in knots if section.start < knot < section.end)})
            for low, high in zip(edges[:-1], edges[1:]):
                loss = loss + _bend_piece(low, high, table, self.state, position, frequency, xp)
        return loss

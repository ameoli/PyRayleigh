"""Distributed power loss in the material coordinate."""

from dataclasses import dataclass

from ..backends import real_arrays
from ..fiber import Fiber


@dataclass(frozen=True)
class FiberLoss:
    """Full section attenuation, with the same convention as PyOFDR.

    A stored Rayleigh realization leaves the distributed coefficient
    unchanged. Loss is integrated over material lengths.
    """

    fiber: Fiber

    def __post_init__(self):
        if not isinstance(self.fiber, Fiber):
            raise TypeError("fiber must be a Fiber")

    def attenuation_integral(self, position, frequency, *, xp=None):
        """A(s, nu) = int_0^s alpha ds, shape position.shape + frequency.shape.

        Finite positions [m] are clipped to the fiber extent. Frequencies
        are absolute Hz. Rates must be finite and nonnegative [1/m].
        """
        xp, arrays = real_arrays(position, frequency,
                                 *(section.attenuation for section in self.fiber.sections), xp=xp)
        position, frequency = arrays[:2]
        position_shape, frequency_shape = position.shape, frequency.shape
        position = xp.reshape(position, position_shape + (1,)*len(frequency_shape))
        frequency = xp.reshape(frequency, (1,)*len(position_shape) + frequency_shape)
        position, frequency = xp.broadcast_arrays(position, frequency)
        loss = xp.zeros_like(frequency)
        for section in self.fiber.sections:
            covered = xp.clip(position - float(section.start), 0., float(section.length))
            loss = loss + covered*section.attenuation_at(frequency, xp=xp)
        return loss

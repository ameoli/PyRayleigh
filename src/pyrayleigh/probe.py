"""Samples at which an interrogator asks for the optical response."""

from dataclasses import dataclass, field

from .backends import real_arrays


@dataclass(frozen=True, eq=False)
class Probe:
    """Absolute optical frequencies [Hz] and optional sample times [s].

    Frequencies are positive and finite physical inputs; times are finite.
    Only real dtype and matching shapes are checked here, so inputs remain
    traceable. The probe stores its own copies of the native arrays.
    """

    frequency: object = field(repr=False)
    time: object = field(default=None, repr=False)

    def __post_init__(self):
        values = (self.frequency,) if self.time is None else (self.frequency, self.time)
        xp, arrays = real_arrays(*values)
        if self.time is not None and arrays[1].shape != arrays[0].shape:
            raise ValueError("time must have the same shape as frequency")
        object.__setattr__(self, "frequency", xp.asarray(arrays[0], copy=True))
        if self.time is not None:
            object.__setattr__(self, "time", xp.asarray(arrays[1], copy=True))

    @property
    def shape(self):
        """Shape of the requested response samples."""
        return self.frequency.shape

    def __repr__(self):
        return f"Probe(shape={self.shape}, time={'None' if self.time is None else 'given'})"

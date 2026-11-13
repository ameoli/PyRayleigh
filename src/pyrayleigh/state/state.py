"""Prescribed axial and thermal state of the fiber."""

from dataclasses import dataclass, field

from .fields import Field

FIELD_NAMES = ("axial_strain", "temperature_change")


@dataclass(frozen=True, kw_only=True)
class FiberState:
    """Fields along the material coordinate and in laboratory time.

    axial_strain is total fiber-axis strain [1], including thermal expansion.
    temperature_change is T - fiber.reference_temperature [K].
    Both inputs are Field objects, defaulting to the empty zero field.
    Native parameters are kept as supplied; the state applies no optical
    response or mechanical transfer model.
    """

    axial_strain: Field = field(default_factory=Field)
    temperature_change: Field = field(default_factory=Field)

    def __post_init__(self):
        for name in FIELD_NAMES:
            if not isinstance(getattr(self, name), Field):
                raise TypeError(f"{name} must be a Field")

    @property
    def is_static(self):
        """True when neither field contains a time history."""
        return all(getattr(self, name).is_static for name in FIELD_NAMES)

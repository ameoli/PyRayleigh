"""Prescribed axial, thermal and curvature fields of the fiber."""

from dataclasses import dataclass, field

from .fields import Field, _field_inputs

FIELD_NAMES = ("axial_strain", "temperature_change", "curvature_x", "curvature_y")


@dataclass(frozen=True, kw_only=True)
class FiberState:
    """Fields along the material coordinate and in laboratory time.

    axial_strain is total fiber-axis strain [1], including thermal expansion.
    temperature_change is T - fiber.reference_temperature [K].
    curvature_x and curvature_y are components in fixed transverse axes [1/m].
    All inputs are Field objects, defaulting to the empty zero field.
    Native parameters are kept as supplied; the state applies no optical
    response or mechanical transfer model.
    """

    axial_strain: Field = field(default_factory=Field)
    temperature_change: Field = field(default_factory=Field)
    curvature_x: Field = field(default_factory=Field)
    curvature_y: Field = field(default_factory=Field)

    def __post_init__(self):
        for name in FIELD_NAMES:
            if not isinstance(getattr(self, name), Field):
                raise TypeError(f"{name} must be a Field")

    @property
    def is_static(self):
        """True when no field contains a time history."""
        return all(getattr(self, name).is_static for name in FIELD_NAMES)

    def at_time(self, time, *, xp=None):
        """Freeze every field at the same scalar laboratory time [s]."""
        terms = tuple(term for name in FIELD_NAMES for term in getattr(self, name).terms)
        xp, _, time = _field_inputs(terms, 0., time, xp)
        return FiberState(**{name: getattr(self, name).at_time(time, xp=xp)
                             for name in FIELD_NAMES})

    def curvature_magnitude(self, position, time=0., *, xp=None):
        """Curvature norm [1/m] at broadcast material positions and lab times.

        The straight case uses zero derivatives, keeping local bend-loss
        queries differentiable where the table gives zero loss.
        """
        terms = self.curvature_x.terms + self.curvature_y.terms
        xp, position, time = _field_inputs(terms, position, time, xp)
        x = self.curvature_x.value(position, time, xp=xp)
        y = self.curvature_y.value(position, time, xp=xp)
        squared = x*x + y*y
        zero = squared == 0.
        safe = xp.where(zero, xp.ones_like(squared), squared)
        return xp.where(zero, xp.zeros_like(squared), xp.sqrt(safe))

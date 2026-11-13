"""The built fiber as a sequence of material sections."""

from dataclasses import dataclass, replace
import math
from numbers import Real

from ..scattering import PointReflector
from .section import Section

SECTION_END_TOLERANCE = 1e-12


@dataclass(frozen=True)
class Fiber:
    """Fixed fiber layout, independent of state and interrogator.

    length is in metres, reference_temperature in K and the optical band
    in absolute Hz. Sections must be ordered and cover [0, length].
    Small boundary round-off is snapped to the neighbouring section.
    Section parameters are retained as supplied, including JAX arrays.
    Point elements have distinct positions in [0, length] and are sorted.
    """

    fiber_id: str
    length: float
    reference_temperature: float
    frequency_min: float
    frequency_max: float
    sections: tuple
    elements: tuple = ()

    def __post_init__(self):
        if not isinstance(self.fiber_id, str):
            raise TypeError("fiber_id must be a string")
        if not self.fiber_id or "\0" in self.fiber_id:
            raise ValueError("fiber_id must be nonempty and contain no NUL characters")
        for name in ("length", "reference_temperature", "frequency_min", "frequency_max"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Real):
                raise TypeError(f"{name} must be a fixed real number")
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.frequency_min >= self.frequency_max:
            raise ValueError("frequency_min must be smaller than frequency_max")
        self._check_sections()
        self._check_elements()

    def _check_sections(self):
        sections = list(self.sections)
        if not sections:
            raise ValueError("a fiber needs at least one section")
        if any(not isinstance(section, Section) for section in sections):
            raise TypeError("sections must contain only Section objects")
        tolerance = SECTION_END_TOLERANCE * self.length
        if (abs(sections[0].start) > tolerance
                or abs(sections[-1].end - self.length) > tolerance):
            raise ValueError("sections must cover [0, length]")
        for index in range(1, len(sections)):
            left, right = sections[index - 1], sections[index]
            if abs(left.end - right.start) > tolerance:
                raise ValueError("sections must be contiguous and in order")
            if right.start != left.end:
                sections[index] = replace(right, start=left.end)
        if sections[0].start != 0.:
            sections[0] = replace(sections[0], start=0.)
        if sections[-1].end != self.length:
            sections[-1] = replace(sections[-1], end=self.length)
        object.__setattr__(self, "sections", tuple(sections))

    def _check_elements(self):
        elements = tuple(self.elements)
        for element in elements:
            if not isinstance(element, PointReflector):
                raise TypeError("elements must contain only PointReflector objects")
            if element.position > self.length:
                raise ValueError("element position lies beyond the fiber length")
        elements = tuple(sorted(elements, key=lambda element: element.position))
        for left, right in zip(elements[:-1], elements[1:]):
            if left.position == right.position:
                raise ValueError("elements must have distinct positions")
        object.__setattr__(self, "elements", elements)

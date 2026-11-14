"""Static phase, group delay and transmission through material sections."""

from dataclasses import dataclass, field
import math
from numbers import Real

from ..backends import real_arrays
from ..fiber import Fiber
from ..state import FiberState
from .loss import FiberLoss

_GUIDE_PARAMETERS = ("frequency_ref", "effective_index", "group_index", "gvd",
                     "strain_phase_factor", "strain_group_factor",
                     "temperature_phase_factor", "temperature_group_factor")


@dataclass(frozen=True)
class OpticalPath:
    """Static path model, without cached numerical data.

    Endpoints are fixed material positions [m]. Frequencies [Hz] and native
    guide/state parameters remain differentiable. Phase uses exp(-i*phi).
    """

    fiber: Fiber
    state: FiberState = field(default_factory=FiberState)

    def __post_init__(self):
        if not isinstance(self.fiber, Fiber):
            raise TypeError("fiber must be a Fiber")
        if not isinstance(self.state, FiberState):
            raise TypeError("state must be a FiberState")
        if not self.state.is_static:
            raise NotImplementedError("OpticalPath requires a static FiberState")

    def _parameters(self, frequency, extra=()):
        parameters = [frequency, *extra]
        if self.fiber.index_perturbation is not None:
            parameters.append(self.fiber.index_perturbation.values)
        for section in self.fiber.sections:
            parameters.extend(getattr(section.optics, name) for name in _GUIDE_PARAMETERS)
        for state_field in (self.state.axial_strain, self.state.temperature_change):
            parameters.extend(term.profile.values for term in state_field.terms)
        return parameters

    def _inputs(self, frequency, xp, extra=()):
        xp, arrays = real_arrays(*self._parameters(frequency, extra), xp=xp)
        return xp, arrays[0]

    def _piece_lengths(self, start, end, xp):
        for value in (start, end):
            if isinstance(value, bool) or not isinstance(value, Real):
                raise TypeError("path endpoints must be fixed real numbers")
            if not math.isfinite(value):
                raise ValueError("path endpoints must be finite")
        if not 0. <= start <= end <= self.fiber.length:
            raise ValueError("path requires 0 <= start <= end <= fiber.length")
        start, end = float(start), float(end)
        for section in self.fiber.sections:
            low, high = max(start, float(section.start)), min(end, float(section.end))
            if low >= high:
                continue
            strain = (self.state.axial_strain.integral(high, xp=xp)
                      - self.state.axial_strain.integral(low, xp=xp))
            temperature = (self.state.temperature_change.integral(high, xp=xp)
                           - self.state.temperature_change.integral(low, xp=xp))
            optics = section.optics
            _, (sp, tp, sg, tg) = real_arrays(optics.strain_phase_factor,
                                             optics.temperature_phase_factor,
                                             optics.strain_group_factor,
                                             optics.temperature_group_factor, xp=xp)
            phase_length = (high - low) + sp*strain + tp*temperature
            group_length = (high - low) + sg*strain + tg*temperature
            if self.fiber.index_perturbation is not None:
                profile = self.fiber.index_perturbation
                index_integral = profile.integral(high, xp=xp) - profile.integral(low, xp=xp)
                _, (phase_index, group_index) = real_arrays(
                    optics.effective_index, optics.group_index, xp=xp)
                phase_length = phase_length + index_integral/phase_index
                group_length = group_length + index_integral/group_index
            yield optics, phase_length, group_length

    def _phase_parts(self, start, end, frequency, xp):
        xp, frequency = self._inputs(frequency, xp)
        carrier, offset = xp.zeros_like(frequency), xp.zeros_like(frequency)
        for optics, phase_length, group_length in self._piece_lengths(start, end, xp):
            beta_ref = optics.propagation_constant(optics.frequency_ref, xp=xp)
            carrier = carrier + beta_ref*phase_length
            offset = offset + optics.propagation_constant_offset(frequency, xp=xp)*group_length
        return xp, carrier, offset

    def one_way_phase(self, start, end, frequency, *, xp=None):
        """Accumulated one-way phase [rad]; a large absolute number."""
        _, carrier, offset = self._phase_parts(start, end, frequency, xp)
        return carrier + offset

    def phase_factor(self, start, end, frequency, round_trip=False, *, xp=None):
        """exp(-i*m*phi), with m=2 for a round trip; carrier and offset kept apart."""
        xp, carrier, offset = self._phase_parts(start, end, frequency, xp)
        multiple = 2. if round_trip else 1.
        i = xp.asarray(-1j, dtype=xp.complex128)
        return (xp.exp(i*xp.astype(multiple*carrier, xp.complex128))
                * xp.exp(i*xp.astype(multiple*offset, xp.complex128)))

    def group_delay(self, start, end, frequency, *, xp=None):
        """One-way d(phi)/d(omega) [s]."""
        xp, frequency = self._inputs(frequency, xp)
        delay = xp.zeros_like(frequency)
        for optics, _, group_length in self._piece_lengths(start, end, xp):
            delay = delay + optics.group_delay_per_length(frequency, xp=xp)*group_length
        return delay

    @property
    def loss(self):
        return FiberLoss(self.fiber, self.state)

    def attenuation_integral(self, position, frequency, *, xp=None):
        """Cumulative distributed power loss A(position, frequency) [1]."""
        return self.loss.attenuation_integral(position, frequency, xp=xp)

    def _transmission(self, start, end, frequency, round_trip, xp):
        xp, frequency = self._inputs(frequency, xp, extra=self.loss._parameters())
        phase = self.phase_factor(start, end, frequency, round_trip, xp=xp)
        loss = (self.attenuation_integral(end, frequency, xp=xp)
                - self.attenuation_integral(start, frequency, xp=xp))
        factor = 1. if round_trip else .5
        envelope = xp.exp(-factor*loss)
        return xp.astype(envelope, xp.complex128)*phase

    def transmission(self, start, end, frequency, *, xp=None):
        """One-way distributed field factor exp(-dA/2)*exp(-i*phi)."""
        return self._transmission(start, end, frequency, False, xp)

    def round_trip(self, start, end, frequency, *, xp=None):
        """Distributed there-and-back field factor exp(-dA)*exp(-2i*phi)."""
        return self._transmission(start, end, frequency, True, xp)


def optical_path(fiber, state=None):
    """Build the static path model; omitted state means the reference fiber."""
    return OpticalPath(fiber, FiberState() if state is None else state)

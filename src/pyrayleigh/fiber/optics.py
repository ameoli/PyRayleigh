"""Reference optical parameters of one guided mode."""

from dataclasses import dataclass

from ..backends import real_arrays
from ..constants import SPEED_OF_LIGHT


@dataclass(frozen=True)
class GuideOptics:
    """Scalar mode parameters at frequency_ref [Hz].

    effective_index and group_index are positive and dimensionless.
    gvd is beta2 [s^2/m]. The expansion keeps terms through beta2.
    Parameters are stored as supplied, including native JAX values.
    """

    frequency_ref: float
    effective_index: float
    group_index: float
    gvd: float = 0.

    def _parameters(self, frequency, xp):
        xp, values = real_arrays(frequency, self.frequency_ref, self.effective_index,
                                 self.group_index, self.gvd, xp=xp)
        frequency, reference, index, group, gvd = values
        if any(value.ndim != 0 for value in values[1:]):
            raise ValueError("guide parameters must be scalars")
        beta_ref = 2. * xp.pi * index * reference / SPEED_OF_LIGHT
        delta_omega = 2. * xp.pi * (frequency - reference)
        return beta_ref, delta_omega, group, gvd

    @property
    def reference_propagation_constant(self):
        """beta at frequency_ref [rad/m]."""
        return self._parameters(self.frequency_ref, None)[0]

    def propagation_constant_offset(self, frequency, *, xp=None):
        """beta(nu)-beta(nu_ref) [rad/m], evaluated without carrier subtraction."""
        _, delta_omega, group, gvd = self._parameters(frequency, xp)
        return group / SPEED_OF_LIGHT * delta_omega + .5 * gvd * delta_omega**2

    def propagation_constant(self, frequency, *, xp=None):
        """beta(nu) [rad/m] at absolute optical frequencies [Hz]."""
        reference, delta_omega, group, gvd = self._parameters(frequency, xp)
        return reference + group / SPEED_OF_LIGHT * delta_omega + .5 * gvd * delta_omega**2

    def group_delay_per_length(self, frequency, *, xp=None):
        """d beta / d omega [s/m]."""
        _, delta_omega, group, gvd = self._parameters(frequency, xp)
        return group / SPEED_OF_LIGHT + gvd * delta_omega

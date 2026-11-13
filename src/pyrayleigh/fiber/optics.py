"""Reference optical parameters of one guided mode."""

from dataclasses import dataclass

from ..backends import real_arrays
from ..constants import SPEED_OF_LIGHT

PHOTOELASTIC_COEFFICIENT = .22
THERMO_OPTIC_COEFFICIENT = 6.5e-6
THERMAL_EXPANSION = 5.5e-7


def _relative_change(axial_strain, temperature_change, strain_factor, temperature_factor, xp):
    _, (strain, temperature, strain_factor, temperature_factor) = real_arrays(
        axial_strain, temperature_change, strain_factor, temperature_factor, xp=xp)
    return strain_factor*strain + temperature_factor*temperature


@dataclass(frozen=True)
class GuideOptics:
    """Scalar mode parameters at frequency_ref [Hz].

    effective_index and group_index are positive and dimensionless.
    gvd is beta2 [s^2/m]. The expansion keeps terms through beta2.
    Strain factors are dimensionless; temperature factors and free thermal
    expansion are in 1/K. Sensitivities use total axial strain, with the
    temperature factors defined at fixed total strain.
    Parameters are stored as supplied, including native JAX values.
    """

    frequency_ref: float
    effective_index: float
    group_index: float
    gvd: float = 0.
    strain_phase_factor: float = 1. - PHOTOELASTIC_COEFFICIENT
    strain_group_factor: float = 1. - PHOTOELASTIC_COEFFICIENT
    temperature_phase_factor: float = (THERMO_OPTIC_COEFFICIENT
                                       + PHOTOELASTIC_COEFFICIENT*THERMAL_EXPANSION)
    temperature_group_factor: float = (THERMO_OPTIC_COEFFICIENT
                                       + PHOTOELASTIC_COEFFICIENT*THERMAL_EXPANSION)
    thermal_expansion: float = THERMAL_EXPANSION

    def __post_init__(self):
        for name in ("frequency_ref", "effective_index", "group_index", "gvd",
                     "strain_phase_factor", "strain_group_factor",
                     "temperature_phase_factor", "temperature_group_factor", "thermal_expansion"):
            _, (value,) = real_arrays(getattr(self, name))
            if value.ndim != 0:
                raise ValueError("guide parameters must be scalars")

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

    def relative_phase_change(self, axial_strain=0., temperature_change=0., *, xp=None):
        """Relative phase change per material length at frequency_ref, to first order."""
        return _relative_change(axial_strain, temperature_change, self.strain_phase_factor,
                                self.temperature_phase_factor, xp)

    def relative_group_delay_change(self, axial_strain=0., temperature_change=0., *, xp=None):
        """Relative group-delay change per material length, to first order."""
        return _relative_change(axial_strain, temperature_change, self.strain_group_factor,
                                self.temperature_group_factor, xp)

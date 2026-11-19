"""Uniform coupled-mode sections and their optical reference planes."""

from dataclasses import dataclass
import math
from numbers import Real

import array_api_compat as apc
import numpy as np

from ..backends import complex_arrays, real_arrays
from ..constants import SPEED_OF_LIGHT


def uniform_bragg_two_port(detuning, coupling, length, *, attenuation=0.):
    """NumPy CMT solution, with detuning and complex coupling in 1/m.

    length [m] is a finite nonnegative scalar. Detuning, coupling and
    background power attenuation [1/m] broadcast. Attenuation is nonnegative.
    delta=beta-beta_B; the grating phase is carried by kappa.
    A'=-(alpha/2+i*delta)*A-i*kappa*B and
    B'=i*conj(kappa)*A+(alpha/2+i*delta)*B, with A travelling right and B left.
    Both ports use this common rotating frame: kappa=0 gives
    t=exp(-(alpha/2+i*delta)*length). Optical carrier phase is not included.
    No material mapping or imposed state is applied here.
    """
    from ..propagation.cascade import TwoPort

    xp, _ = complex_arrays(detuning, coupling, length, attenuation)
    if not apc.is_numpy_namespace(xp):
        raise NotImplementedError('uniform_bragg_two_port currently supports NumPy only')
    _, (delta, width, alpha) = real_arrays(detuning, length, attenuation, xp=xp)
    _, (kappa,) = complex_arrays(coupling, xp=xp)
    if width.ndim != 0 or not np.isfinite(width) or width < 0.:
        raise ValueError('length must be a finite nonnegative scalar')
    if not np.all(np.isfinite(alpha)) or np.any(alpha < 0.):
        raise ValueError('attenuation must be finite and nonnegative')
    if not np.all(np.isfinite(delta)) or not np.all(np.isfinite(kappa)):
        raise ValueError('detuning and coupling must be finite')

    with np.errstate(over='raise', invalid='raise', divide='raise', under='ignore'):
        d, k, loss = np.broadcast_arrays(delta*width, kappa*width, .5j*alpha*width)
        if np.any(alpha != 0.):
            d = d-loss
            u = np.sqrt(np.abs(k)**2-d*d)
            # Principal square root has nonnegative real part.
            decay = np.exp(-u)
            c = .5*(1.+decay*decay)
            s = np.ones_like(u)
            np.divide(-np.expm1(-2.*u), 2.*u, out=s, where=u != 0.)
            numerator = decay
        else:
            magnitude = np.abs(k)
            squared = (magnitude-np.abs(d))*(magnitude+np.abs(d))
            u = np.sqrt(np.abs(squared))
            stop = squared >= 0.
            c = np.array(np.cos(u))
            s = np.array(np.sinc(u/np.pi))
            numerator = np.ones_like(u)

            # Divide out cosh in the stop band, avoiding overflow for strong gratings.
            inside = u[stop]
            ratio = np.ones_like(inside)
            np.divide(np.tanh(inside), inside, out=ratio, where=inside != 0.)
            c[stop], s[stop] = 1., ratio
            decay = np.exp(-inside)
            numerator[stop] = 2.*decay/(1.+decay*decay)

        denominator = c+1j*d*s
        transmission = numerator/denominator
        result = TwoPort(-1j*np.conj(k)*s/denominator, -1j*k*s/denominator,
            transmission, transmission)
    result.validate_passivity()
    return result


@dataclass(frozen=True)
class BraggGrating:
    """Uniform grating on the material interval [start, start+length] [m].

    bragg_wavelength [m] fixes beta_B through the supplied reference guide.
    coupling [1/m] may be complex; its phase is referenced to start.
    Optical parameters remain native scalars; values are checked by two_port.
    Geometry and coupling are fixed in the material coordinate. Uniform
    strain, temperature and index change affect propagation, without changing
    coupling.
    segment_length [m] optionally enables midpoint sampling of continuous
    loads in the exact fiber solver. It does not change the Rayleigh grid.
    """

    start: float
    length: float
    bragg_wavelength: object
    coupling: object
    segment_length: float | None = None

    def __post_init__(self):
        for name in ('start', 'length'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Real):
                raise TypeError(f'{name} must be a fixed real number')
            if not math.isfinite(value) or value < 0. or (name == 'length' and value == 0.):
                raise ValueError('start must be nonnegative and length must be positive and finite')
        if not math.isfinite(self.end) or self.end <= self.start:
            raise ValueError('grating end must be finite and larger than start')
        if self.segment_length is not None:
            if isinstance(self.segment_length, bool) or not isinstance(self.segment_length, Real):
                raise TypeError('segment_length must be a fixed real number or None')
            if not math.isfinite(self.segment_length) or self.segment_length <= 0.:
                raise ValueError('segment_length must be finite and positive')
        _, (wavelength,) = real_arrays(self.bragg_wavelength)
        _, (coupling,) = complex_arrays(self.coupling)
        if wavelength.ndim != 0 or coupling.ndim != 0:
            raise ValueError('grating wavelength and coupling must be scalars')

    @property
    def end(self):
        return float(self.start)+float(self.length)

    def _parameters(self):
        return self.bragg_wavelength, self.coupling

    def two_port(self, frequency, optics, *, axial_strain=0., temperature_change=0.,
                 attenuation=0., index_change=0.):
        """NumPy optical fields at start/end, for absolute frequency [Hz].

        optics is the uniform reference GuideOptics. Strain [1] and T-T_ref
        [K] and index_change [1] must be uniform scalars. The Bragg material
        period stays fixed; carrier phase uses that unperturbed frame.
        attenuation is spatially uniform background power loss [1/m],
        broadcast with frequency. It may include tabulated bend loss.
        Index changes follow OpticalPath, without extra interface scattering.
        Input-to-start propagation is separate.
        """
        from ..fiber import GuideOptics
        from ..propagation.cascade import TwoPort

        if not isinstance(optics, GuideOptics):
            raise TypeError('optics must be a GuideOptics')
        parameters = (frequency, *self._parameters(), optics.frequency_ref,
            optics.effective_index, optics.group_index, optics.gvd,
            axial_strain, temperature_change, attenuation, index_change, optics.strain_phase_factor,
            optics.strain_group_factor, optics.temperature_phase_factor, optics.temperature_group_factor)
        xp, arrays = complex_arrays(*parameters)
        if not apc.is_numpy_namespace(xp):
            raise NotImplementedError('BraggGrating.two_port currently supports NumPy only')
        _, (frequency, wavelength) = real_arrays(frequency, self.bragg_wavelength, xp=xp)
        _, (strain, temperature, index) = real_arrays(axial_strain, temperature_change, index_change, xp=xp)
        if strain.ndim != 0 or temperature.ndim != 0 or index.ndim != 0:
            raise ValueError('grating strain, temperature and index change must be uniform scalars')
        if any(not np.all(np.isfinite(value)) for value in arrays):
            raise ValueError('grating response parameters must be finite')
        if np.any(frequency <= 0.) or wavelength <= 0.:
            raise ValueError('frequency and Bragg wavelength must be positive')
        if optics.frequency_ref <= 0. or optics.effective_index <= 0. or optics.group_index <= 0.:
            raise ValueError('guide reference frequency and indices must be positive')

        with np.errstate(over='raise', invalid='raise', divide='raise', under='ignore'):
            bragg_frequency = SPEED_OF_LIGHT/wavelength
            delta_omega = 2*np.pi*(frequency-bragg_frequency)
            detuning = (optics.group_delay_per_length(bragg_frequency)*delta_omega
                + .5*optics.gvd*delta_omega**2)
            phase_change = optics.relative_phase_change(strain, temperature, xp=xp)
            group_change = optics.relative_group_delay_change(strain, temperature, xp=xp)
            if index != 0.:
                phase_change = phase_change+index/optics.effective_index
                group_change = group_change+index/optics.group_index
            detuning = (detuning+optics.reference_propagation_constant*phase_change
                +optics.propagation_constant_offset(frequency)*group_change)
            envelope = uniform_bragg_two_port(detuning, self.coupling, self.length, attenuation=attenuation)
            carrier = (np.exp(-1j*optics.reference_propagation_constant*self.length)
                *np.exp(-1j*optics.propagation_constant_offset(bragg_frequency)*self.length))
            result = TwoPort(envelope.r_left, envelope.r_right*carrier**2,
                envelope.t_left_to_right*carrier, envelope.t_right_to_left*carrier)
        result.validate_passivity()
        return result

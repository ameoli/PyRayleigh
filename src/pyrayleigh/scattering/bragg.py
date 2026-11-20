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


def piecewise_bragg_two_port(detuning, coupling, lengths, *, attenuation=0.,
                              denominator_tolerance=64*np.finfo(np.float64).eps):
    """NumPy CMT cascade in a common envelope frame, without optical carrier.

    lengths is a nonempty vector of finite nonnegative segment lengths [m].
    Detuning, complex coupling and background power loss [1/m] broadcast.
    Their first axis is the segment axis (size 1 or len(lengths)); trailing
    axes are spectral/batch axes. Scalars apply to every segment. For N
    segments and F frequencies use shapes (N, F), (N, 1), etc.
    Coupling phases refer to the same rotating frame across all segments.
    Each segment checks passivity. The cascade cutoff is forwarded to star;
    it is not an accuracy guarantee near a narrow resonance.
    """
    from ..propagation.cascade import _check_tolerance, star

    xp, _ = complex_arrays(detuning, coupling, lengths, attenuation)
    if not apc.is_numpy_namespace(xp):
        raise NotImplementedError('piecewise_bragg_two_port currently supports NumPy only')
    _, (delta, widths, alpha) = real_arrays(detuning, lengths, attenuation, xp=xp)
    _, (kappa,) = complex_arrays(coupling, xp=xp)
    _check_tolerance(denominator_tolerance)
    if widths.ndim != 1 or widths.size == 0 or not np.all(np.isfinite(widths)) or np.any(widths < 0.):
        raise ValueError('lengths must be a nonempty vector of finite nonnegative values')
    if any(not np.all(np.isfinite(value)) for value in (delta, kappa, alpha)):
        raise ValueError('detuning, coupling and attenuation must be finite')
    if np.any(alpha < 0.):
        raise ValueError('attenuation must be nonnegative')
    delta, kappa, alpha = np.broadcast_arrays(delta, kappa, alpha)
    shape = (widths.size,)+delta.shape[1:] if delta.ndim else (widths.size,)
    delta, kappa, alpha = (np.broadcast_to(value, shape) for value in (delta, kappa, alpha))
    result = None
    for index, width in enumerate(widths):
        block = uniform_bragg_two_port(delta[index], kappa[index], width, attenuation=alpha[index])
        result = block if result is None else star(result, block,
            denominator_tolerance=denominator_tolerance)
    return result


@dataclass(frozen=True)
class BraggGrating:
    """Grating on the material interval [start, start+length] [m].

    bragg_wavelength [m] fixes beta_B through the supplied reference guide.
    coupling [1/m] may be complex; its phase is referenced to start.
    Optical parameters remain native scalars; values are checked by two_port.
    Geometry and coupling are fixed in the material coordinate. Uniform
    strain, temperature and index change affect propagation, without changing
    coupling.
    segment_length [m] optionally enables midpoint sampling of continuous
    loads in the exact fiber solver. It does not change the Rayleigh grid.
    apodization is an optional nonnegative dimensionless SpatialProfile in
    absolute material coordinates. It multiplies coupling and is zero outside
    its support. A continuous envelope requires segment_length.
    coupling_phase is an optional SpatialProfile in unwrapped radians on the
    same coordinates. It multiplies coupling by exp(1j*phase), with phase zero
    outside its support. Continuous phase profiles also require segment_length.
    """

    start: float
    length: float
    bragg_wavelength: object
    coupling: object
    segment_length: float | None = None
    apodization: object = None
    coupling_phase: object = None

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
        from ..state import SpatialProfile
        for name in ('apodization', 'coupling_phase'):
            profile = getattr(self, name)
            if profile is not None and not isinstance(profile, SpatialProfile):
                raise TypeError(f'{name} must be a SpatialProfile or None')

    @property
    def end(self):
        return float(self.start)+float(self.length)

    def _parameters(self):
        parameters = self.bragg_wavelength, self.coupling
        for profile in (self.apodization, self.coupling_phase):
            if profile is not None:
                parameters += (profile.values,)
        return parameters

    def two_port(self, frequency, optics, *, axial_strain=0., temperature_change=0.,
                 attenuation=0., index_change=0., denominator_tolerance=64*np.finfo(np.float64).eps):
        """NumPy optical fields at start/end, for absolute frequency [Hz].

        optics is the uniform reference GuideOptics. Strain [1] and T-T_ref
        [K] and index_change [1] must be uniform scalars. The Bragg material
        period stays fixed; carrier phase uses that unperturbed frame.
        attenuation is spatially uniform background power loss [1/m],
        broadcast with frequency. It may include tabulated bend loss.
        Index changes follow OpticalPath, without extra interface scattering.
        Coupling profiles use the same piecewise/midpoint rule as the fiber solver.
        denominator_tolerance is forwarded to their CMT cascade.
        Input-to-start propagation is separate.
        """
        from ..fiber import GuideOptics
        from ..propagation.cascade import TwoPort, _TOLERANCE, _check_tolerance

        if not isinstance(optics, GuideOptics):
            raise TypeError('optics must be a GuideOptics')
        _check_tolerance(denominator_tolerance)
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
        if self.apodization is not None and np.any(self.apodization.values < 0.):
            raise ValueError('apodization must be nonnegative')
        if np.any(frequency <= 0.) or wavelength <= 0.:
            raise ValueError('frequency and Bragg wavelength must be positive')
        if optics.frequency_ref <= 0. or optics.effective_index <= 0. or optics.group_index <= 0.:
            raise ValueError('guide reference frequency and indices must be positive')

        segments = 1
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
            if self.apodization is None and self.coupling_phase is None:
                envelope = uniform_bragg_two_port(detuning, self.coupling, self.length, attenuation=attenuation)
            else:
                from ..propagation.exact import _midpoint_load, _uniform_value
                profiles = (self.apodization, self.coupling_phase)
                edges = sorted({self.start, self.end, *(float(knot) for profile in profiles
                    if profile is not None for knot in profile.positions
                    if self.start < knot < self.end)})
                if self.segment_length is not None:
                    refined = []
                    for left, right in zip(edges[:-1], edges[1:]):
                        count = max(1, math.ceil((right-left)/self.segment_length))
                        refined.extend(np.linspace(left, right, count+1)[:-1])
                    edges = np.asarray([*refined, edges[-1]])
                    if np.any(np.diff(edges) <= 0.):
                        raise ValueError('Bragg segment edges must be distinct')
                samples = []
                for profile, default in zip(profiles, (1., 0.)):
                    if profile is None:
                        values = np.full(len(edges)-1, default)
                    elif self.segment_length is None:
                        values = np.array([_uniform_value(profile, a, b) for a, b in zip(edges[:-1], edges[1:])])
                    else:
                        values = np.array([_midpoint_load(profile, a, b)[0] for a, b in zip(edges[:-1], edges[1:])])
                    samples.append(values)
                weights, phases = samples
                coupling = self.coupling*weights
                if np.any(phases != 0.):
                    coupling = coupling*np.exp(1j*phases)
                if np.all(coupling == coupling[0]):
                    envelope = uniform_bragg_two_port(detuning, coupling[0], self.length,
                        attenuation=attenuation)
                else:
                    segments = len(weights)
                    coupling = coupling.reshape((len(weights),)+(1,)*frequency.ndim)
                    envelope = piecewise_bragg_two_port(detuning[None, ...], coupling, np.diff(edges),
                        attenuation=np.asarray(attenuation)[None, ...], denominator_tolerance=denominator_tolerance)
            carrier = (np.exp(-1j*optics.reference_propagation_constant*self.length)
                *np.exp(-1j*optics.propagation_constant_offset(bragg_frequency)*self.length))
            result = TwoPort(envelope.r_left, envelope.r_right*carrier**2,
                envelope.t_left_to_right*carrier, envelope.t_right_to_left*carrier)
        result.validate_passivity(tolerance=_TOLERANCE*segments)
        return result

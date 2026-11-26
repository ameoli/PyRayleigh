"""Uniform coupled-mode sections and their optical reference planes."""

from dataclasses import dataclass, replace
import math
from numbers import Real

import array_api_compat as apc
import numpy as np

from ..backends import complex_arrays, real_arrays
from ..constants import SPEED_OF_LIGHT


def _native_uniform_bragg(delta, kappa, width, alpha, xp):
    from ..propagation.cascade import TwoPort

    delta, kappa, alpha = xp.broadcast_arrays(delta, kappa, alpha)
    valid = (xp.isfinite(delta) & xp.isfinite(kappa) & xp.isfinite(alpha)
        & (alpha >= 0.) & xp.isfinite(width) & (width >= 0.))
    delta, kappa, alpha, width = (xp.where(valid, value, xp.zeros_like(value))
        for value in (delta, kappa, alpha, width))
    with np.errstate(over='ignore', invalid='ignore', divide='ignore', under='ignore'):
        d = xp.astype(delta*width, xp.complex128)-.5j*xp.astype(alpha*width, xp.complex128)
        k = kappa*width
        q = xp.astype(xp.real(k*xp.conj(k)), xp.complex128)-d*d
        valid = valid & xp.isfinite(d) & xp.isfinite(k) & xp.isfinite(q)
        d, k, q = (xp.where(valid, value, xp.zeros_like(value)) for value in (d, k, q))
        small = xp.abs(q) < 1e-4
        local = xp.where(small, q, xp.zeros_like(q))
        # Entire functions of q avoid the square-root singularity at a band edge.
        c0 = 1.+local*(.5+local*(1./24.+local*(1./720.+local/40320.)))
        s0 = 1.+local*(1./6.+local*(1./120.+local*(1./5040.+local/362880.)))
        root = xp.sqrt(xp.where(small, xp.ones_like(q), q))
        decay = xp.exp(-root)
        c = xp.where(small, c0, .5*(1.+decay*decay))
        s = xp.where(small, s0, -xp.expm1(-2.*root)/(2.*root))
        numerator = xp.where(small, xp.ones_like(q), decay)
        denominator = c+1j*d*s
        t = numerator/denominator
        fields = (-1j*xp.conj(k)*s/denominator, -1j*k*s/denominator, t, t)
        for value in fields:
            valid = valid & xp.isfinite(value)
        valid = valid & TwoPort(*fields).passivity_mask()
        fields = tuple(xp.where(valid, value, xp.full_like(value, float('nan'))) for value in fields)
    return TwoPort(*fields), valid


def uniform_bragg_two_port(detuning, coupling, length, *, attenuation=0., return_validity=False):
    """CMT solution, with detuning and complex coupling in 1/m.

    length [m] is a finite nonnegative scalar. Detuning, coupling and
    background power attenuation [1/m] broadcast. Attenuation is nonnegative.
    delta=beta-beta_B; the grating phase is carried by kappa.
    A'=-(alpha/2+i*delta)*A-i*kappa*B and
    B'=i*conj(kappa)*A+(alpha/2+i*delta)*B, with A travelling right and B left.
    Both ports use this common rotating frame: kappa=0 gives
    t=exp(-(alpha/2+i*delta)*length). Optical carrier phase is not included.
    No material mapping or imposed state is applied here.
    NumPy retains numerical exceptions. Other namespaces keep native fields;
    invalid parameters/arithmetic or failed passivity produce NaN samples.
    return_validity=True also returns the sample mask, usable inside jit.
    """
    from ..propagation.cascade import TwoPort

    xp, _ = complex_arrays(detuning, coupling, length, attenuation)
    _, (delta, width, alpha) = real_arrays(detuning, length, attenuation, xp=xp)
    _, (kappa,) = complex_arrays(coupling, xp=xp)
    if not isinstance(return_validity, (bool, np.bool_)):
        raise TypeError('return_validity must be a fixed bool')
    if width.ndim != 0:
        raise ValueError('length must be a scalar')
    if not apc.is_numpy_namespace(xp):
        result, valid = _native_uniform_bragg(delta, kappa, width, alpha, xp)
        return (result, valid) if return_validity else result
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
    return (result, xp.ones_like(result.r_left, dtype=xp.bool)) if return_validity else result


def piecewise_bragg_two_port(detuning, coupling, lengths, *, attenuation=0.,
                              denominator_tolerance=64*np.finfo(np.float64).eps,
                              return_validity=False):
    """CMT cascade in a common envelope frame, without optical carrier.

    lengths is a nonempty vector of finite nonnegative segment lengths [m].
    Detuning, complex coupling and background power loss [1/m] broadcast.
    Their first axis is the segment axis (size 1 or len(lengths)); trailing
    axes are spectral/batch axes. Scalars apply to every segment. For N
    segments and F frequencies use shapes (N, F), (N, 1), etc.
    Coupling phases refer to the same rotating frame across all segments.
    Each segment checks passivity. The cascade cutoff is forwarded to star;
    it is not an accuracy guarantee near a narrow resonance.
    The fixed segment count allows native tracing. return_validity=True also
    returns the combined section/cascade mask. NumPy keeps its exceptions;
    other namespaces return NaN fields at invalid samples, visiting all sections.
    """
    from ..propagation.cascade import _check_tolerance, star

    xp, _ = complex_arrays(detuning, coupling, lengths, attenuation)
    _, (delta, widths, alpha) = real_arrays(detuning, lengths, attenuation, xp=xp)
    _, (kappa,) = complex_arrays(coupling, xp=xp)
    _check_tolerance(denominator_tolerance)
    if not isinstance(return_validity, (bool, np.bool_)):
        raise TypeError('return_validity must be a fixed bool')
    if widths.ndim != 1 or widths.shape[0] == 0:
        raise ValueError('lengths must be a nonempty vector of finite nonnegative values')
    if apc.is_numpy_namespace(xp):
        if not np.all(np.isfinite(widths)) or np.any(widths < 0.):
            raise ValueError('lengths must be a nonempty vector of finite nonnegative values')
        if any(not np.all(np.isfinite(value)) for value in (delta, kappa, alpha)):
            raise ValueError('detuning, coupling and attenuation must be finite')
        if np.any(alpha < 0.):
            raise ValueError('attenuation must be nonnegative')
    delta, kappa, alpha = xp.broadcast_arrays(delta, kappa, alpha)
    count = widths.shape[0]
    shape = (count,)+delta.shape[1:] if delta.ndim else (count,)
    delta, kappa, alpha = (xp.broadcast_to(value, shape) for value in (delta, kappa, alpha))
    result, valid = None, None
    for index in range(count):
        block, local_valid = uniform_bragg_two_port(delta[index, ...], kappa[index, ...], widths[index],
            attenuation=alpha[index, ...], return_validity=True)
        if result is None:
            result, valid = block, local_valid
        else:
            result, joined = star(result, block, denominator_tolerance=denominator_tolerance, return_validity=True)
            valid = valid & local_valid & joined
    return (result, valid) if return_validity else result


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

    def with_phase_shift(self, phase_shift=np.pi, *, shift_position=None):
        """Copy the grating, adding a phase step [rad] to its stored phase.

        shift_position [m] is a fixed absolute material coordinate, strictly
        inside the grating; the default is its midpoint. Repeated calls add
        steps without wrapping phases or changing the existing interpolant.
        Each step inserts its boundary into the CMT sampling mesh.
        """
        from ..fiber.index_profiles import add_profiles

        xp, _ = complex_arrays(*self._parameters(), phase_shift)
        _, (phase,) = real_arrays(phase_shift, xp=xp)
        step = phase_shifted_bragg(self.start, self.length, self.bragg_wavelength,
            self.coupling, phase, shift_position=shift_position,
            segment_length=self.segment_length, apodization=self.apodization).coupling_phase
        if self.coupling_phase is not None:
            step = add_profiles(self.coupling_phase, step, xp=xp)
        return replace(self, coupling_phase=step)

    def two_port(self, frequency, optics, *, axial_strain=0., temperature_change=0.,
                 attenuation=0., index_change=0., denominator_tolerance=64*np.finfo(np.float64).eps,
                 return_validity=False):
        """Optical fields at start/end, for absolute frequency [Hz].

        optics is the uniform reference GuideOptics. Strain [1] and T-T_ref
        [K] and index_change [1] must be uniform scalars. The Bragg material
        period stays fixed; carrier phase uses that unperturbed frame.
        attenuation is spatially uniform background power loss [1/m],
        broadcast with frequency. It may include tabulated bend loss.
        Index changes follow OpticalPath, without extra interface scattering.
        Coupling profiles use the same piecewise/midpoint rule as the fiber solver.
        denominator_tolerance is forwarded to their CMT cascade.
        Input-to-start propagation is separate.
        Native namespaces currently support uniform gratings. Invalid native
        samples return NaN fields; return_validity=True also returns the mask.
        NumPy keeps its numerical exceptions and support for stored profiles.
        """
        from ..fiber import GuideOptics
        from ..propagation.cascade import TwoPort, _TOLERANCE, _check_tolerance

        if not isinstance(optics, GuideOptics):
            raise TypeError('optics must be a GuideOptics')
        _check_tolerance(denominator_tolerance)
        if not isinstance(return_validity, (bool, np.bool_)):
            raise TypeError('return_validity must be a fixed bool')
        parameters = (frequency, *self._parameters(), optics.frequency_ref,
            optics.effective_index, optics.group_index, optics.gvd,
            axial_strain, temperature_change, attenuation, index_change, optics.strain_phase_factor,
            optics.strain_group_factor, optics.temperature_phase_factor, optics.temperature_group_factor)
        xp, arrays = complex_arrays(*parameters)
        is_numpy = apc.is_numpy_namespace(xp)
        _, (frequency, wavelength) = real_arrays(frequency, self.bragg_wavelength, xp=xp)
        _, (strain, temperature, index) = real_arrays(axial_strain, temperature_change, index_change, xp=xp)
        if strain.ndim != 0 or temperature.ndim != 0 or index.ndim != 0:
            raise ValueError('grating strain, temperature and index change must be uniform scalars')
        if is_numpy:
            if any(not np.all(np.isfinite(value)) for value in arrays):
                raise ValueError('grating response parameters must be finite')
            if self.apodization is not None and np.any(self.apodization.values < 0.):
                raise ValueError('apodization must be nonnegative')
            if np.any(frequency <= 0.) or wavelength <= 0.:
                raise ValueError('frequency and Bragg wavelength must be positive')
            if optics.frequency_ref <= 0. or optics.effective_index <= 0. or optics.group_index <= 0.:
                raise ValueError('guide reference frequency and indices must be positive')
        else:
            if self.apodization is not None or self.coupling_phase is not None:
                raise NotImplementedError('native BraggGrating.two_port currently supports uniform gratings only')
            valid = xp.ones_like(frequency, dtype=xp.bool)
            for value in arrays:
                valid = valid & xp.isfinite(value)
            valid = (valid & (frequency > 0.) & (wavelength > 0.) & (optics.frequency_ref > 0.)
                & (optics.effective_index > 0.) & (optics.group_index > 0.))

        segments = 1
        length = self.length if is_numpy else float(self.length)
        policy = 'raise' if is_numpy else 'ignore'
        with np.errstate(over=policy, invalid=policy, divide=policy, under='ignore'):
            bragg_frequency = SPEED_OF_LIGHT/wavelength
            delta_omega = 2*xp.pi*(frequency-bragg_frequency)
            detuning = (optics.group_delay_per_length(bragg_frequency, xp=xp)*delta_omega
                + .5*optics.gvd*delta_omega**2)
            phase_change = optics.relative_phase_change(strain, temperature, xp=xp)
            group_change = optics.relative_group_delay_change(strain, temperature, xp=xp)
            if not is_numpy or index != 0.:
                phase_change = phase_change+index/optics.effective_index
                group_change = group_change+index/optics.group_index
            beta_ref = (optics.reference_propagation_constant if is_numpy
                else optics._parameters(optics.frequency_ref, xp)[0])
            detuning = (detuning+beta_ref*phase_change
                +optics.propagation_constant_offset(frequency, xp=xp)*group_change)
            if self.apodization is None and self.coupling_phase is None:
                envelope, local_valid = uniform_bragg_two_port(detuning, self.coupling, length,
                    attenuation=attenuation, return_validity=True)
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
            carrier_ref = beta_ref if is_numpy else xp.astype(beta_ref, xp.complex128)
            carrier_offset = optics.propagation_constant_offset(bragg_frequency, xp=xp)
            if not is_numpy:
                carrier_offset = xp.astype(carrier_offset, xp.complex128)
            carrier = (xp.exp(-1j*carrier_ref*length)*xp.exp(-1j*carrier_offset*length))
            result = TwoPort(envelope.r_left, envelope.r_right*carrier**2,
                envelope.t_left_to_right*carrier, envelope.t_right_to_left*carrier)
        if is_numpy:
            result.validate_passivity(tolerance=_TOLERANCE*segments)
            valid = xp.ones_like(result.r_left, dtype=xp.bool)
        else:
            valid = valid & local_valid & result.passivity_mask()
            fields = (result.r_left, result.r_right, result.t_left_to_right, result.t_right_to_left)
            result = TwoPort(*(xp.where(valid, value, xp.full_like(value, float('nan'))) for value in fields))
        return (result, valid) if return_validity else result


def linear_chirped_bragg(start, length, bragg_wavelength, coupling, detuning_slope, *,
                         segment_length, apodization=None):
    """Build a grating with a linear effective-detuning slope [1/m^2].

    bragg_wavelength fixes the reference at the material midpoint. With
    u=s-start, phi=detuning_slope*(u*u-length*u) gives
    delta_eff=delta+detuning_slope*(u-length/2), with zero phase at the left.
    The quadratic phase is stored as a sampled SpatialProfile. Rebuild with
    smaller segment_length [m] to refine both this profile and the CMT mesh.
    Geometry is fixed; phase values retain the native parameter namespace.
    """
    from ..state import SpatialProfile

    if segment_length is None:
        raise ValueError('a chirped grating requires segment_length')
    grating = BraggGrating(start, length, bragg_wavelength, coupling,
        segment_length=segment_length, apodization=apodization)
    xp, _ = complex_arrays(*grating._parameters(), detuning_slope)
    _, (slope,) = real_arrays(detuning_slope, xp=xp)
    if slope.ndim != 0:
        raise ValueError('detuning_slope must be a scalar')
    if apc.is_numpy_namespace(xp) and not np.isfinite(slope):
        raise ValueError('detuning_slope must be finite')
    count = max(2, math.ceil(float(length)/float(segment_length)))
    positions = np.linspace(float(start), grating.end, count+1)
    if np.any(np.diff(positions) <= 0.):
        raise ValueError('chirped grating knots must be distinct')
    fractions = xp.asarray(np.linspace(0., 1., count+1).tolist(), dtype=xp.float64)
    phase = slope*(float(length)*fractions)*(float(length)*(fractions-1.))
    return replace(grating, coupling_phase=SpatialProfile(positions, phase))


def gaussian_apodization(start, length, sigma, *, sample_spacing):
    """Sample a unit-peak Gaussian envelope, with width sigma [m].

    Coordinates are absolute material metres. The Gaussian is truncated to
    [start, start+length]; endpoint values follow the Gaussian, and the
    profile is zero outside. The fixed grid includes the material midpoint.
    Rebuild with smaller sample_spacing to refine this linear profile, and
    refine the grating's segment_length for the CMT calculation.
    """
    from ..state import SpatialProfile

    for name, value in (('start', start), ('length', length), ('sample_spacing', sample_spacing)):
        if isinstance(value, bool) or not isinstance(value, Real):
            raise TypeError(f'{name} must be a fixed real number')
        if not math.isfinite(value) or value < 0. or (name != 'start' and value == 0.):
            raise ValueError('start must be nonnegative; length and sample_spacing must be positive and finite')
    end = float(start)+float(length)
    if not math.isfinite(end) or end <= start:
        raise ValueError('profile end must be finite and larger than start')
    xp, (width,) = real_arrays(sigma)
    if width.ndim != 0:
        raise ValueError('sigma must be a scalar')
    if apc.is_numpy_namespace(xp) and (not np.isfinite(width) or width <= 0.):
        raise ValueError('sigma must be finite and positive')
    count = max(2, math.ceil(float(length)/float(sample_spacing)))
    count += count % 2
    positions = np.linspace(float(start), end, count+1)
    if np.any(np.diff(positions) <= 0.):
        raise ValueError('Gaussian profile knots must be distinct')
    fractions = xp.asarray(np.linspace(0., 1., count+1).tolist(), dtype=xp.float64)
    with np.errstate(over='ignore', under='ignore', divide='ignore', invalid='ignore'):
        values = xp.exp(-.5*(float(length)*(fractions-.5)/width)**2)
    values = xp.where(xp.isfinite(width) & (width > 0.), values, xp.full_like(values, float('nan')))
    return SpatialProfile(positions, values)


def phase_shifted_bragg(start, length, bragg_wavelength, coupling, phase_shift=np.pi, *,
                        shift_position=None, segment_length=None, apodization=None):
    """Build a grating with one jump in coupling phase [rad], pi by default.

    shift_position [m] is a fixed absolute material coordinate, defaulting
    to the midpoint. It must be strictly inside the grating. The right
    part has coupling*exp(1j*phase_shift), in the same frame as the left.
    The step is stored directly; segment_length is optional for uniform
    loads, and can sample an optional continuous apodization profile.
    """
    from ..state import SpatialProfile

    grating = BraggGrating(start, length, bragg_wavelength, coupling,
        segment_length=segment_length, apodization=apodization)
    position = grating.start+.5*(grating.end-grating.start) if shift_position is None else shift_position
    if isinstance(position, bool) or not isinstance(position, Real):
        raise TypeError('shift_position must be a fixed real number')
    if not math.isfinite(position) or not grating.start < position < grating.end:
        raise ValueError('shift_position must be strictly inside the grating')
    xp, _ = complex_arrays(*grating._parameters(), phase_shift)
    _, (phase,) = real_arrays(phase_shift, xp=xp)
    if phase.ndim != 0:
        raise ValueError('phase_shift must be a scalar')
    if apc.is_numpy_namespace(xp) and not np.isfinite(phase):
        raise ValueError('phase_shift must be finite')
    zero = xp.zeros_like(phase)
    profile = SpatialProfile([grating.start, position, position, grating.end], xp.stack((zero, zero, phase, phase)))
    return replace(grating, coupling_phase=profile)

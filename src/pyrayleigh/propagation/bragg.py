"""Native scalar cascade inside a Bragg interval."""

import math

from ..backends import complex_arrays, real_arrays
from ..constants import SPEED_OF_LIGHT
from ..scattering import uniform_bragg_two_port
from ..state import Field, FieldTerm
from .cascade import TwoPort, _TOLERANCE, star
from .loss import _curvature_crossings


def _interval_load(field, start, end, xp):
    value, change = xp.asarray(0., dtype=xp.float64), xp.asarray(0., dtype=xp.float64)
    for term in field.terms:
        profile = term.profile
        _, (values,) = real_arrays(profile.values, xp=xp)
        for index, (left, right) in enumerate(zip(profile.positions[:-1], profile.positions[1:])):
            if left <= start and end <= right and left < right:
                delta = values[index+1]-values[index]
                fraction = float(((start-left)+.5*(end-start))/(right-left))
                value = value+values[index]+fraction*delta
                change = change+delta*float((end-start)/(right-left))
                break
    return value, change


def _native_bragg_block(grating, frequency, optics, state, denominator_tolerance, *,
                        attenuation, index_perturbation, scatterers, xp):
    profiles = (index_perturbation, grating.apodization, grating.coupling_phase)
    fields = (state.axial_strain, state.temperature_change)+tuple(
        Field() if profile is None else Field((FieldTerm(profile),)) for profile in profiles)
    table = optics.bend_loss
    if table is not None:
        fields = fields+(state.curvature_x, state.curvature_y)
    edges = sorted({grating.start, grating.end, *(position for position, _ in scatterers),
        *(float(knot) for field in fields for term in field.terms for knot in term.profile.positions
            if grating.start < knot < grating.end)})
    _, (wavelength, reference, neff, ng, gvd, sp, sg, tp, tg, alpha) = real_arrays(
        grating.bragg_wavelength, optics.frequency_ref, optics.effective_index, optics.group_index,
        optics.gvd, optics.strain_phase_factor, optics.strain_group_factor,
        optics.temperature_phase_factor, optics.temperature_group_factor, attenuation, xp=xp)
    _, (coupling,) = complex_arrays(grating.coupling, xp=xp)
    bragg_frequency = SPEED_OF_LIGHT/wavelength
    beta_ref = 2.*xp.pi*neff*reference/SPEED_OF_LIGHT
    omega_b = 2.*xp.pi*(bragg_frequency-reference)
    beta_b_offset = ng/SPEED_OF_LIGHT*omega_b+.5*gvd*omega_b**2
    omega = 2.*xp.pi*(frequency-bragg_frequency)
    detuning = (ng/SPEED_OF_LIGHT+gvd*omega_b)*omega+.5*gvd*omega**2
    offset = 2.*xp.pi*(frequency-reference)
    beta_offset = ng/SPEED_OF_LIGHT*offset+.5*gvd*offset**2

    def carrier(distance):
        distance = xp.asarray(distance, dtype=xp.complex128)
        return (xp.exp(-1j*xp.astype(beta_ref, xp.complex128)*distance)
            *xp.exp(-1j*xp.astype(beta_b_offset, xp.complex128)*distance))

    zero = xp.zeros_like(frequency, dtype=xp.complex128)
    one = xp.ones_like(zero)
    result = TwoPort(zero, zero, one, one)
    valid = xp.isfinite(wavelength) & (wavelength > 0.)
    if grating.apodization is not None:
        _, (weights,) = real_arrays(grating.apodization.values, xp=xp)
        valid = valid & xp.all(weights >= 0.)
    scatterer_index, pieces = 0, 0
    for left, right in zip(edges[:-1], edges[1:]):
        loads = [_interval_load(field, left, right, xp) for field in fields]
        middle, changes = xp.stack([item[0] for item in loads]), xp.stack([item[1] for item in loads])
        if grating.segment_length is None:
            valid = valid & xp.all(changes == 0.)
        count = 1 if grating.segment_length is None else max(1, math.ceil((right-left)/grating.segment_length))
        boundaries = xp.asarray([0., 1.], dtype=xp.float64)
        if table is not None:
            x, y = middle[5]-.5*changes[5], middle[6]-.5*changes[6]
            upper = float(table.curvatures[-1])
            valid = (valid & (x*x+y*y <= upper**2)
                & ((x+changes[5])**2+(y+changes[6])**2 <= upper**2))
            if grating.segment_length is not None:
                boundaries = _curvature_crossings(x, y, changes[5], changes[6], table.curvatures, xp)
        for index in range(boundaries.shape[0]-1):
            start, stop = boundaries[index], boundaries[index+1]
            for part in range(count):
                a = start+(stop-start)*(part/count)
                b = start+(stop-start)*((part+1)/count)
                local = middle+(.5*(a+b)-.5)*changes
                strain, temperature, dn = local[0], local[1], local[2]
                weight = xp.asarray(1., dtype=xp.float64) if grating.apodization is None else local[3]
                kappa = coupling*xp.astype(weight, xp.complex128)*xp.exp(1j*xp.astype(local[4], xp.complex128))
                delta = (detuning+beta_ref*(sp*strain+tp*temperature+dn/neff)
                    +beta_offset*(sg*strain+tg*temperature+dn/ng))
                loss = alpha
                if table is not None:
                    squared = local[5]**2+local[6]**2
                    magnitude = xp.sqrt(xp.where(squared > 0., squared, xp.ones_like(squared)))
                    magnitude = xp.where(squared > 0., magnitude, xp.zeros_like(magnitude))
                    loss = loss+table.value(magnitude, frequency, xp=xp)
                block, local_valid = uniform_bragg_two_port(delta, kappa, (b-a)*float(right-left),
                    attenuation=loss, return_validity=True)
                result, connected = star(result, block, denominator_tolerance=denominator_tolerance,
                    return_validity=True)
                valid = valid & local_valid & connected
                pieces += 1
        while scatterer_index < len(scatterers) and scatterers[scatterer_index][0] == right:
            position, block = scatterers[scatterer_index]
            # Local point coefficients enter the common grating frame here.
            phase = carrier(float(position-grating.start))**2
            block = TwoPort(block.r_left*phase, block.r_right*xp.conj(phase),
                block.t_left_to_right, block.t_right_to_left)
            result, connected = star(result, block, denominator_tolerance=denominator_tolerance,
                return_validity=True)
            valid = valid & connected
            scatterer_index += 1
    phase = carrier(float(grating.length))
    result = TwoPort(result.r_left, result.r_right*phase**2,
        result.t_left_to_right*phase, result.t_right_to_left*phase)
    return result, valid & result.passivity_mask(tolerance=_TOLERANCE*(pieces+len(scatterers)))

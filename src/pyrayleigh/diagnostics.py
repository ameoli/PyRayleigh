"""Optional exact/direct comparison for a fixed scalar fiber."""

from dataclasses import dataclass, field

import array_api_compat as apc
import numpy as np

from .backends import complex_arrays, real_arrays
from .propagation.cascade import _TOLERANCE
from .propagation.direct import direct_response_consistent
from .propagation.exact import exact_response
from .response import Response


@dataclass(frozen=True, eq=False)
class DiagnosticResponse:
    """Optical fields, with residual = total - direct in both ports.

    The residual belongs to the same local block model, not to the legacy
    approximation. Power also needs its interference with the direct field.
    """

    total: Response
    direct: Response
    metadata: dict = field(default_factory=dict)
    residual: Response = field(init=False)

    def __post_init__(self):
        if not isinstance(self.total, Response) or not isinstance(self.direct, Response):
            raise TypeError("total and direct must be Response objects")
        if self.total.probe is not self.direct.probe:
            raise ValueError("total and direct must share the same Probe")
        xp = apc.array_namespace(self.total.reflection, self.total.transmission,
            self.direct.reflection, self.direct.transmission, self.total.probe.frequency)
        if not apc.is_numpy_namespace(xp):
            raise NotImplementedError("DiagnosticResponse currently supports NumPy only")
        object.__setattr__(self, "metadata", dict(self.metadata))
        residual = Response(self.total.probe,
            reflection=self.total.reflection-self.direct.reflection,
            transmission=self.total.transmission-self.direct.transmission)
        object.__setattr__(self, "residual", residual)

    @property
    def probe(self):
        return self.total.probe

    @property
    def shape(self):
        return self.total.shape

    @property
    def reflection_interference(self):
        """Signed power term 2*real(direct_r*conj(residual_r))."""
        return 2*np.real(self.direct.reflection*np.conj(self.residual.reflection))

    @property
    def transmission_interference(self):
        """Signed power term 2*real(direct_t*conj(residual_t))."""
        return 2*np.real(self.direct.transmission*np.conj(self.residual.transmission))


def compare_responses(response, reference, *, field_floor=1e-12):
    """NumPy field, power and wrapped-phase errors, separately for each port.

    Both responses must have identical frequency and optional time samples.
    relative_l2 uses max(norm(reference), field_floor) as denominator. Phase
    errors use only samples where both amplitudes exceed field_floor, and
    are None when no sample qualifies. No interpolation or weighting is used.
    The caller chooses the reference; this alone does not prove convergence.
    """
    if not isinstance(response, Response) or not isinstance(reference, Response):
        raise TypeError('response and reference must be Response objects')
    parameters = [response.reflection, response.transmission, reference.reflection,
        reference.transmission, response.probe.frequency, reference.probe.frequency, field_floor]
    parameters.extend(item.probe.time for item in (response, reference) if item.probe.time is not None)
    xp, arrays = complex_arrays(*parameters)
    if not apc.is_numpy_namespace(xp):
        raise NotImplementedError('compare_responses currently supports NumPy only')
    _, (floor,) = real_arrays(field_floor, xp=xp)
    if floor.ndim != 0 or not np.isfinite(floor) or floor <= 0.:
        raise ValueError('field_floor must be a finite positive scalar')
    if any(not np.all(np.isfinite(value)) for value in arrays):
        raise ValueError('comparison inputs must be finite')
    if not np.array_equal(response.probe.frequency, reference.probe.frequency):
        raise ValueError('responses must have identical frequency samples')
    time, reference_time = response.probe.time, reference.probe.time
    if (time is None) != (reference_time is None) or (time is not None and not np.array_equal(time, reference_time)):
        raise ValueError('responses must have identical time samples')

    report = {}
    with np.errstate(over='raise', invalid='raise', divide='raise', under='ignore'):
        for port in ('reflection', 'transmission'):
            value, target = getattr(response, port), getattr(reference, port)
            amplitude, reference_amplitude = np.abs(value), np.abs(target)
            difference = np.abs(value-target)
            error = np.hypot.reduce(difference.reshape(-1), initial=0.)
            scale = np.hypot.reduce(reference_amplitude.reshape(-1), initial=0.)
            mask = (amplitude > floor) & (reference_amplitude > floor)
            phase = np.angle((value[mask]/amplitude[mask])*np.conj(target[mask]/reference_amplitude[mask]))
            report[port] = {'relative_l2': float(error/max(scale, float(floor))),
                'max_field_error': float(np.max(difference, initial=0.)),
                'max_power_error': float(np.max(np.abs(amplitude**2-reference_amplitude**2), initial=0.)),
                'max_phase_error_rad': float(np.max(np.abs(phase))) if phase.size else None,
                'phase_samples': int(phase.size)}
    return report


def compute_response_diagnostics(fiber, probe, state=None, *, loss_convention,
                                 denominator_tolerance=_TOLERANCE):
    """Compute exact, direct-consistent and residual fields on the same input.

    This runs the two scalar static NumPy calculations separately. Normal
    compute_response calls do not allocate these extra fields. Metadata is a
    short model summary, without copying the realization or inventing a seed.
    """
    total = exact_response(fiber, probe, state, loss_convention=loss_convention,
        denominator_tolerance=denominator_tolerance)
    direct = direct_response_consistent(fiber, probe, state, loss_convention=loss_convention,
        denominator_tolerance=denominator_tolerance)
    metadata = {"fiber_id": fiber.fiber_id, "loss_convention": loss_convention,
        "model": "scalar_static", "backend": "numpy", "dtype": "complex128",
        "boundary": "matched_right", "denominator_tolerance": float(denominator_tolerance),
        "rayleigh_cells": 0 if fiber.rayleigh is None else fiber.rayleigh.cell_count}
    if fiber.gratings:
        metadata.update(bragg_gratings=len(fiber.gratings), direct_scope='one_reflection_per_block')
        points = tuple(int(sum(grating.start < point.position < grating.end
            for point in fiber.elements)) for grating in fiber.gratings)
        if any(points):
            metadata['bragg_internal_points'] = points
        if fiber.rayleigh is not None:
            inside = tuple(int(np.count_nonzero((fiber.rayleigh.positions > grating.start)
                & (fiber.rayleigh.positions < grating.end))) for grating in fiber.gratings)
            if any(inside):
                metadata['bragg_internal_rayleigh'] = inside
        if any(grating.segment_length is not None for grating in fiber.gratings):
            metadata['bragg_segment_lengths_m'] = tuple(None if grating.segment_length is None
                else float(grating.segment_length) for grating in fiber.gratings)
    return DiagnosticResponse(total, direct, metadata)

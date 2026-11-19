"""Compare coupled Rayleigh grids after installing PyRayleigh.

The finest grid is a numerical reference, not an analytic solution.
This is one realization, not an ensemble convergence test.
"""

from dataclasses import replace

import numpy as np

from pyrayleigh import (Fiber, FiberState, Field, FieldTerm, GuideOptics, Probe,
    Section, SpatialProfile, compute_response_diagnostics, refine_rayleigh_from_normals)
from pyrayleigh.constants import SPEED_OF_LIGHT


def coupled_grids():
    frequency = SPEED_OF_LIGHT/1550e-9
    length = .08
    guide = GuideOptics(frequency, 1.47, 1.47)
    fiber = Fiber('coupled', length, 293.15, frequency-5e9, frequency+5e9,
        (Section(0., length, guide, 0., .1),))
    fiber = fiber.with_rayleigh(123, cell_length=.01)
    fibers = [fiber]
    rng = np.random.Generator(np.random.PCG64(124))
    for _ in range(5):
        parent = fibers[-1]
        normals = rng.standard_normal((parent.rayleigh.cell_count, 2))
        sample = refine_rayleigh_from_normals(parent, normals,
            reference_frequency=frequency)
        fibers.append(replace(parent, rayleigh=sample))
    return fibers


def relative_error(value, reference):
    scale = np.linalg.norm(reference)
    if scale == 0.:
        return np.nan
    return float(np.linalg.norm(value-reference)/scale)


def run():
    fibers = coupled_grids()
    probe = Probe(np.linspace(fibers[0].frequency_min, fibers[0].frequency_max, 129))
    strain = SpatialProfile([.027, .028, .039, .040], [0., 1e-5, 1e-5, 0.])
    state = FiberState(axial_strain=Field((FieldTerm(strain),)))
    unloaded, loaded = [], []
    for fiber in fibers:
        unloaded.append(compute_response_diagnostics(fiber, probe,
            loss_convention='background'))
        loaded.append(compute_response_diagnostics(fiber, probe, state,
            loss_convention='background'))
    return fibers, probe, unloaded, loaded


def main():
    fibers, probe, unloaded, loaded = run()
    print('One frozen realization, 10 GHz band, background attenuation = 0.')
    print('Errors are complex-field L2 ratios over the band, relative to the finest grid.')
    print('No phase alignment is fitted between spectra. nan means zero reference norm.')
    for name, results in (('unloaded', unloaded), ('local strain = 10 microstrain', loaded)):
        reference = results[-1]
        print('\n'+name)
        print('cells   max ds [m]   exact r error   direct r error   exact t error   residual/exact r')
        for fiber, result in zip(fibers, results):
            width = np.max(fiber.rayleigh.cell_lengths())
            er = relative_error(result.total.reflection, reference.total.reflection)
            dr = relative_error(result.direct.reflection, reference.direct.reflection)
            et = relative_error(result.total.transmission, reference.total.transmission)
            residual = np.linalg.norm(result.residual.reflection)/np.linalg.norm(result.total.reflection)
            print(f'{fiber.rayleigh.cell_count:5d}   {width:.4e}   {er:.6e}   {dr:.6e}   {et:.6e}   {residual:.6e}')

    change = [b.total.reflection-a.total.reflection for a, b in zip(unloaded, loaded)]
    print('\nStrain-induced change of the exact reflected field')
    print('cells   relative error in delta r')
    for fiber, value in zip(fibers, change):
        print(f'{fiber.rayleigh.cell_count:5d}   {relative_error(value, change[-1]):.6e}')
    print('\nThe last row has zero error by construction; extend the hierarchy to check it.')
    print('Refinement fixes an unloaded phase frame at the band centre, not the full spectrum.')
    print('Rayleigh coefficients are reused unchanged for both states; exact/direct share depletion.')


if __name__ == '__main__':
    main()

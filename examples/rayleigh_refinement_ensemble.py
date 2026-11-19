"""Repeat the coupled-grid comparison on several frozen Rayleigh samples.

Run after installing PyRayleigh. This finite ensemble gives descriptive
errors against its finest grids, not confidence bounds or an exact limit.
"""

import argparse

import numpy as np

from rayleigh_refinement import run


def run_ensemble(*, samples=16, levels=7, seed=124):
    if samples < 2 or levels < 2:
        raise ValueError('at least two samples and two grid levels are required')
    fields = []
    for stream in np.random.SeedSequence(seed).spawn(samples):
        initial_seed, bridge_seed = stream.generate_state(2, dtype=np.uint64)
        fibers, _, unloaded, loaded = run(seed=int(initial_seed),
            refinement_seed=int(bridge_seed), levels=levels)
        before = np.stack([result.total.reflection for result in unloaded])
        after = np.stack([result.total.reflection for result in loaded])
        fields.append(np.stack((before, after, after-before)))
    cells = np.array([fiber.rayleigh.cell_count for fiber in fibers])
    return cells, np.stack(fields)


def field_errors(fields):
    """fields has shape (samples, levels, frequencies), without phase fitting."""
    reference = fields[:, -1, :]
    norms = np.linalg.norm(reference, axis=-1)
    if np.any(norms == 0.) or not np.all(np.isfinite(norms)):
        raise ValueError('reference fields must have finite, nonzero norms')
    errors = np.linalg.norm(fields-reference[:, None, :], axis=-1)
    changes = np.linalg.norm(np.diff(fields, axis=1), axis=-1)
    scale = np.linalg.norm(norms)
    rms = np.linalg.norm(errors, axis=0)/scale
    p90 = np.quantile(errors/norms[:, None], .9, axis=0)
    increments = np.linalg.norm(changes, axis=0)/scale
    return rms, p90, increments


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--samples', type=int, default=16)
    parser.add_argument('--levels', type=int, default=7)
    parser.add_argument('--seed', type=int, default=124)
    args = parser.parse_args()
    if args.samples < 2 or args.levels < 2 or args.seed < 0:
        parser.error('samples and levels must be >= 2; seed must be nonnegative')
    print(f'{args.samples} realizations, {args.levels} levels, root seed {args.seed}.', flush=True)
    cells, fields = run_ensemble(samples=args.samples, levels=args.levels, seed=args.seed)
    print('Exact reflected fields on 10 GHz, before/after 10 microstrain; fixed samples within each hierarchy.')
    print('RMS = sqrt(sum ||error||^2 / sum ||finest field||^2) over realizations.')
    print('p90 is the descriptive quantile of individual relative errors; no confidence interval.')
    print('The adjacent-grid increment uses the same pooled finest-field normalization.')
    for name, data in zip(('unloaded r', 'loaded r', 'strain-induced delta r'), np.moveaxis(fields, 1, 0)):
        rms, p90, increments = field_errors(data)
        print('\n'+name)
        print('cells   RMS vs finest   p90 vs finest   RMS increment from previous')
        for index, count in enumerate(cells):
            step = '---' if index == 0 else f'{increments[index-1]:.6e}'
            print(f'{count:5d}   {rms[index]:.6e}   {p90[index]:.6e}   {step}')
    print('\nThe finest-grid error is zero by construction, but its adjacent increment is not.')
    print('Increase --levels and --samples to examine reference-grid and sampling uncertainty.')


if __name__ == '__main__':
    main()

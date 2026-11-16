"""Mechanical strain transferred from a host structure to a bonded fiber."""

from ..backends import real_arrays


def cox_strain(position, start, end, host_strain, beta, *, xp=None):
    """Cox shear-lag strain for a uniformly loaded bond [start, end].

    Positions are in m, beta > 0 in 1/m, and host_strain is dimensionless.
    Inputs are finite real values with end > start; only dtype is checked.
    Arrays broadcast. The result is mechanical strain, zero outside the bond;
    independent bonds add. Position/endpoint derivatives exclude bond edges.
    """
    xp, (position, start, end, host, beta) = real_arrays(
        position, start, end, host_strain, beta, xp=xp)
    length = end - start
    left = xp.clip(position - start, 0., length)
    right = length - left
    factor = (-xp.expm1(-beta*left))*(-xp.expm1(-beta*right))/(1. + xp.exp(-beta*length))
    return host*factor

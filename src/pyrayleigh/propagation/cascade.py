"""Scalar two-port fields and composition in native array namespaces."""

from dataclasses import dataclass, field
from numbers import Real

import array_api_compat as apc
import numpy as np

from ..backends import complex_arrays


_TOLERANCE = 64*np.finfo(np.float64).eps


def _check_tolerance(value):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
            or not np.isfinite(value) or value < 0.):
        raise ValueError("tolerance must be a finite nonnegative real number")


class NumericalResonanceError(FloatingPointError):
    """A cascade denominator is below the requested numerical cutoff."""

    def __init__(self, index, denominator, tolerance, *, frequency=None, position=None):
        self.index = index
        self.denominator = complex(denominator)
        self.magnitude = abs(self.denominator)
        self.tolerance = float(tolerance)
        self.frequency = frequency
        self.position = position
        message = (f"star denominator at index {index} has magnitude "
                   f"{self.magnitude:.6g} <= {self.tolerance:.6g}")
        if frequency is not None:
            message += f" at {frequency:.12g} Hz"
        if position is not None:
            message += f", material position {position:.12g} m"
        super().__init__(message)


@dataclass(frozen=True, eq=False)
class TwoPort:
    """Copied native complex128 fields with a common broadcast shape.

    The matrix is [[r_left, t_right_to_left], [t_left_to_right, r_right]].
    NumPy fields must be finite and are read-only. Other namespaces keep
    only dtype/shape checks, allowing traced parameters. Numerical passivity
    checks currently require NumPy.
    """

    r_left: object = field(repr=False)
    r_right: object = field(repr=False)
    t_left_to_right: object = field(repr=False)
    t_right_to_left: object = field(repr=False)

    def __post_init__(self):
        names = ("r_left", "r_right", "t_left_to_right", "t_right_to_left")
        xp, arrays = complex_arrays(*(getattr(self, name) for name in names))
        is_numpy = apc.is_numpy_namespace(xp)
        arrays = xp.broadcast_arrays(*arrays)
        for name, value in zip(names, arrays):
            if is_numpy and not np.all(np.isfinite(value)):
                raise ValueError(f"{name} must be finite")
            value = xp.asarray(value, dtype=xp.complex128, copy=True)
            if is_numpy:
                value.setflags(write=False)
            object.__setattr__(self, name, value)

    @property
    def shape(self):
        return self.r_left.shape

    def scattering_matrix(self):
        """Matrix axes are the final two axes: shape + (2, 2)."""
        xp, _ = complex_arrays(self.r_left)
        return xp.stack((xp.stack((self.r_left, self.t_right_to_left), axis=-1),
                         xp.stack((self.t_left_to_right, self.r_right), axis=-1)), axis=-2)

    def validate_passivity(self, tolerance=_TOLERANCE):
        """Reject matrices whose largest singular value exceeds 1+tolerance."""
        _check_tolerance(tolerance)
        if not apc.is_numpy_namespace(apc.array_namespace(self.r_left)):
            raise NotImplementedError("validate_passivity currently supports NumPy only")
        largest = np.linalg.svd(self.scattering_matrix(), compute_uv=False)[..., 0]
        invalid = ~np.isfinite(largest) | (largest > 1.+tolerance)
        if np.any(invalid):
            index = tuple(int(i) for i in np.argwhere(invalid)[0])
            raise ValueError(f"non-passive TwoPort at index {index}: sigma_max={largest[index]:.6g}")

    def __repr__(self):
        return f"TwoPort(shape={self.shape})"


def star(left, right, *, denominator_tolerance=_TOLERANCE, return_validity=False):
    """Connect left's right port to right's left port.

    Validate physical components before composing them. The tolerance is an
    absolute refusal cutoff, not a regularization or an accuracy guarantee.
    NumPy raises at the cutoff, including singular closed cavities.
    Other namespaces mark invalid samples with NaN in all four fields.
    return_validity=True also returns a native mask for finite arithmetic
    and denominators above the cutoff; this is not a passivity check.
    Derivatives are meaningful only for valid samples away from the cutoff.
    """
    if not isinstance(left, TwoPort) or not isinstance(right, TwoPort):
        raise TypeError("star requires two TwoPort objects")
    _check_tolerance(denominator_tolerance)
    if not isinstance(return_validity, (bool, np.bool_)):
        raise TypeError("return_validity must be a fixed bool")
    xp, _ = complex_arrays(left.r_left, right.r_left)
    is_numpy = apc.is_numpy_namespace(xp)
    values = (left.r_left, left.r_right, left.t_left_to_right, left.t_right_to_left,
        right.r_left, right.r_right, right.t_left_to_right, right.t_right_to_left)
    policy = 'raise' if is_numpy else 'ignore'
    with np.errstate(over=policy, invalid=policy, divide=policy, under='ignore'):
        denominator = xp.asarray(1.-left.r_right*right.r_left)
        valid = xp.isfinite(denominator) & (xp.abs(denominator) > float(denominator_tolerance))
        if is_numpy:
            invalid = np.abs(denominator) <= denominator_tolerance
            if np.any(invalid):
                index = tuple(int(i) for i in np.argwhere(invalid)[0])
                raise NumericalResonanceError(index, denominator[index], denominator_tolerance)
        else:
            for value in values:
                valid = valid & xp.isfinite(value)
            values = tuple(xp.where(valid, value, xp.zeros_like(value)) for value in values)
            denominator = xp.where(valid, denominator, xp.ones_like(denominator))
        r_a, rp_a, t_a, tp_a, r_b, rp_b, t_b, tp_b = values
        fields = (r_a+tp_a*r_b*t_a/denominator, rp_b+t_b*rp_a*tp_b/denominator,
            t_b*t_a/denominator, tp_a*tp_b/denominator)
        if not is_numpy:
            for value in fields:
                valid = valid & xp.isfinite(value)
            fields = tuple(xp.where(valid, value, xp.full_like(value, float('nan'))) for value in fields)
        result = TwoPort(*fields)
    return (result, valid) if return_validity else result

import math

import numpy as np
import pytest

from pyrayleigh import Fiber, GuideOptics, Section, units

OPTICS = GuideOptics(193414489032258.06, 1.4682, 1.4682)


def make_fiber(sections, **metadata):
    values = dict(fiber_id="spool-1", length=2., reference_temperature=293.15,
                  frequency_min=1.9e14, frequency_max=2e14, sections=sections)
    values.update(metadata)
    return Fiber(**values)


def section(start, end, attenuation=0., rayleigh=1e-8):
    return Section(start, end, OPTICS, attenuation, rayleigh)


def test_single_section_fiber():
    part = section(0., 2.)
    fiber = make_fiber([part])
    assert fiber.sections == (part,)
    assert fiber.sections[0] is part
    assert fiber.length == 2.
    assert fiber.reference_temperature == 293.15
    assert fiber.frequency_min == 1.9e14


def test_input_section_list_is_not_retained():
    parts = [section(0., 1.), section(1., 2.)]
    fiber = make_fiber(parts)
    parts.clear()
    assert isinstance(fiber.sections, tuple)
    assert len(fiber.sections) == 2
    assert fiber.sections[0].end == fiber.sections[1].start
    assert sum(part.length for part in fiber.sections) == fiber.length


@pytest.mark.parametrize("metadata,error", [
    ({"fiber_id": ""}, ValueError), ({"fiber_id": "spool\0"}, ValueError),
    ({"fiber_id": 1}, TypeError), ({"length": 0.}, ValueError),
    ({"length": float("nan")}, ValueError), ({"reference_temperature": 0.}, ValueError),
    ({"frequency_min": -1.}, ValueError), ({"frequency_max": float("inf")}, ValueError),
    ({"frequency_min": 2e14}, ValueError), ({"frequency_min": 3e14}, ValueError),
    ({"length": True}, TypeError), ({"length": np.array(2.)}, TypeError),
])
def test_fixed_fiber_metadata(metadata, error):
    with pytest.raises(error):
        make_fiber([section(0., 2.)], **metadata)


@pytest.mark.parametrize("parts", [
    [], [section(.1, 2.)], [section(0., 1.9)], [section(0., 2.1)],
    [section(0., .9), section(1., 2.)],
    [section(0., 1.1), section(1., 2.)],
    [section(1., 2.), section(0., 1.)],
])
def test_incomplete_or_disordered_sections_are_rejected(parts):
    with pytest.raises(ValueError):
        make_fiber(parts)


def test_sections_must_have_the_right_type():
    with pytest.raises(TypeError, match="Section"):
        make_fiber([None])


def test_boundary_roundoff_is_snapped_without_changing_inputs():
    alpha = np.array(4.6e-5)
    rayleigh = np.array(1e-8)
    left = section(5e-13, .3, alpha, rayleigh)
    right = section(.1 + .2, 1. - 5e-13, alpha, rayleigh)
    fiber = make_fiber([left, right], length=1.)
    assert fiber.sections[0].start == 0.
    assert fiber.sections[0].end == fiber.sections[1].start == .3
    assert fiber.sections[1].end == 1.
    assert left.start == 5e-13
    assert right.start == .1 + .2
    assert right.end == 1. - 5e-13
    for part in fiber.sections:
        assert part.attenuation is alpha
        assert part.rayleigh_coefficient is rayleigh
        assert part.optics is OPTICS


def test_snapping_cannot_leave_an_empty_section():
    with pytest.raises(ValueError):
        make_fiber([section(0., 1.), section(1. - 5e-13, 1.)], length=1.)


def test_jax_nested_section_parameters_remain_differentiable():
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp
    previous = jax.config.x64_enabled
    jax.config.update("jax_enable_x64", True)
    try:
        def power(losses_db_per_km):
            parts = [section(5e-10, 1000., units.attenuation_from_db_per_km(losses_db_per_km[0])),
                     section(1000. + 5e-10, 2000. - 5e-10,
                             units.attenuation_from_db_per_km(losses_db_per_km[1]))]
            fiber = make_fiber(parts, length=2000.)
            integrated_loss = sum(part.attenuation_at(193414489032258.06) * part.length
                                  for part in fiber.sections)
            return jnp.exp(-integrated_loss)

        losses = jnp.asarray([.2, .4], dtype=jnp.float64)
        value = jax.jit(power)(losses)
        gradient = jax.jit(jax.grad(power))(losses)
        hessian = jax.jit(jax.hessian(power))(losses)
        expected = .8709635899560807
        scale = math.log(10.) / 10.
        np.testing.assert_allclose(value, expected, rtol=1e-13, atol=0.)
        np.testing.assert_allclose(gradient, np.full(2, -scale*expected), rtol=1e-13, atol=0.)
        np.testing.assert_allclose(hessian, np.full((2, 2), scale**2*expected), rtol=1e-13, atol=0.)
    finally:
        jax.config.update("jax_enable_x64", previous)

from dataclasses import replace
import math

import numpy as np
import pytest

from pyrayleigh import (BendLossTable, Fiber, GuideOptics, RayleighRealization, Section,
                        lossless_reflector, rayleigh_from_normals, units)

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
    assert fiber.elements == ()
    assert fiber.rayleigh is None


@pytest.mark.parametrize("frequencies", [[1.95e14], [1.9e14, 2e14], [1.8e14, 2.1e14]])
def test_section_bend_table_covers_the_declared_band(frequencies):
    table = BendLossTable([10., 20.], frequencies, np.ones((2, len(frequencies)))*.01)
    guide = replace(OPTICS, bend_loss=table)
    fiber = make_fiber([Section(0., 2., guide, 0., 1e-8)])
    assert fiber.sections[0].optics.bend_loss is table


@pytest.mark.parametrize("frequencies", [[1.91e14, 2e14], [1.9e14, 1.99e14]])
@pytest.mark.parametrize("table_in_first_section", [True, False])
def test_each_section_bend_table_must_cover_the_whole_fiber_band(
        frequencies, table_in_first_section):
    table = BendLossTable([10., 20.], frequencies, np.ones((2, 2))*.01)
    guide = replace(OPTICS, bend_loss=table)
    guides = [guide, OPTICS] if table_in_first_section else [OPTICS, guide]
    parts = [Section(0., 1., guides[0], 0., 1e-8), Section(1., 2., guides[1], 0., 1e-8)]
    with pytest.raises(ValueError, match="bend-loss table.*fiber band"):
        make_fiber(parts)


def test_snapping_and_rayleigh_generation_retain_the_same_bend_table():
    table = BendLossTable([10., 20.], [1.95e14], [[.01], [.1]])
    guide = replace(OPTICS, bend_loss=table)
    part = Section(5e-13, 2. - 5e-13, guide, 0., 1e-8)
    fiber = make_fiber([part])
    assert fiber.sections[0].start == 0.
    assert fiber.sections[0].end == 2.
    generated = fiber.with_rayleigh(27, cell_length=.5)
    assert generated.sections[0].optics.bend_loss is table
    assert part.start == 5e-13
    plain = make_fiber([section(0., 2.)]).with_rayleigh(27, cell_length=.5)
    np.testing.assert_array_equal(generated.rayleigh.coefficients, plain.rayleigh.coefficients)


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


def test_elements_are_sorted_without_changing_the_input_objects():
    points = [lossless_reflector(position, .04) for position in (2., 0., 1.)]
    original = tuple(points)
    fiber = make_fiber([section(0., 1.), section(1., 2.)], elements=points)
    assert isinstance(fiber.elements, tuple)
    assert [point.position for point in points] == [2., 0., 1.]
    assert all(actual is expected for actual, expected in
               zip(fiber.elements, (original[1], original[2], original[0])))
    points.clear()
    assert len(fiber.elements) == 3
    assert fiber.elements[0].r_left is original[1].r_left
    assert fiber.elements[-1].t_left_to_right is original[0].t_left_to_right


@pytest.mark.parametrize("position", [2.1, np.nextafter(2., np.inf)])
def test_elements_beyond_the_fiber_are_rejected(position):
    with pytest.raises(ValueError, match="beyond"):
        make_fiber([section(0., 2.)], elements=[lossless_reflector(position, .04)])


@pytest.mark.parametrize("position", [0., 1., 2.])
def test_coincident_elements_are_rejected(position):
    points = [lossless_reflector(position, .04), lossless_reflector(position, .01)]
    with pytest.raises(ValueError, match="distinct"):
        make_fiber([section(0., 2.)], elements=points)


def test_nearby_elements_are_not_snapped_together():
    points = [lossless_reflector(position, .04) for position in
              (np.nextafter(1., 2.), 1.)]
    fiber = make_fiber([section(0., 2.)], elements=points)
    assert fiber.elements[0] is points[1]
    assert fiber.elements[1] is points[0]
    assert fiber.elements[0].position < fiber.elements[1].position


@pytest.mark.parametrize("element", [None, .5, section(0., 2.)])
def test_elements_must_be_point_reflectors(element):
    with pytest.raises(TypeError, match="PointReflector"):
        make_fiber([section(0., 2.)], elements=[element])


def test_attached_rayleigh_keeps_the_supplied_sample_and_partial_cell():
    sample = RayleighRealization([.01j, .02j], [.5, 1.75], [0., 1.5, 2.])
    fiber = make_fiber([section(0., 1.), section(1., 2.)], rayleigh=sample)
    assert fiber.rayleigh is sample
    assert fiber.rayleigh.coefficients is sample.coefficients
    np.testing.assert_array_equal(fiber.rayleigh.cell_lengths(), [1.5, .5])


@pytest.mark.parametrize("extent", [1.9, 2.1, np.nextafter(2., 0.), np.nextafter(2., np.inf)])
def test_rayleigh_must_cover_the_exact_fiber_length(extent):
    sample = RayleighRealization([.01j], [0.], [0., extent])
    with pytest.raises(ValueError, match="cover exactly"):
        make_fiber([section(0., 2.)], rayleigh=sample)


@pytest.mark.parametrize("sample", [0., {}, lossless_reflector(1., .04)])
def test_rayleigh_requires_a_realization(sample):
    with pytest.raises(TypeError, match="RayleighRealization"):
        make_fiber([section(0., 2.)], rayleigh=sample)


def test_with_rayleigh_returns_a_copy_and_can_replace_a_stored_sample():
    point = lossless_reflector(1., .04)
    base = make_fiber([section(0., 2.)], elements=[point])
    built = base.with_rayleigh(7, cell_length=1.5)
    changed = built.with_rayleigh(8, cell_length=1.5)
    assert base.rayleigh is None
    assert built is not base and changed is not built
    assert built.sections[0] is base.sections[0]
    assert built.elements[0] is point
    assert built.rayleigh.extent == base.length
    assert changed.rayleigh is not built.rayleigh
    assert not np.array_equal(changed.rayleigh.coefficients, built.rayleigh.coefficients)


def test_changing_section_rates_keeps_the_stored_sample():
    built = make_fiber([section(0., 2.)]).with_rayleigh(7, cell_length=1.5)
    changed = replace(built, sections=[section(0., 2., rayleigh=4e-8)])
    assert changed.rayleigh is built.rayleigh


def test_jax_attached_rayleigh_preserves_section_rate_derivatives():
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp
    previous = jax.config.x64_enabled
    jax.config.update("jax_enable_x64", True)
    try:
        def local_reflection(rate):
            fiber = make_fiber([section(0., 2., rayleigh=rate*1e-8)])
            sample = rayleigh_from_normals(fiber, [[1., 2.], [3., -1.]],
                                           cell_length=1.5, xp=jnp)
            fiber = replace(fiber, rayleigh=sample)
            return jnp.real(fiber.rayleigh.coefficients[0])

        rate = jnp.asarray(2., dtype=jnp.float64)
        value = jax.jit(local_reflection)(rate)
        first = jax.jit(jax.grad(local_reflection))(rate)
        second = jax.jit(jax.grad(jax.grad(local_reflection)))(rate)
        expected = math.sqrt(1.5e-8)
        np.testing.assert_allclose(value, expected, rtol=1e-14, atol=0.)
        np.testing.assert_allclose(first, expected/4., rtol=1e-13, atol=0.)
        np.testing.assert_allclose(second, -expected/16., rtol=1e-13, atol=0.)
    finally:
        jax.config.update("jax_enable_x64", previous)


def test_jax_nested_element_parameters_remain_differentiable():
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp
    previous = jax.config.x64_enabled
    jax.config.update("jax_enable_x64", True)
    try:
        def local_reflected_field(reflectance):
            points = [lossless_reflector(1.5, .01),
                      lossless_reflector(.5, reflectance, .3)]
            fiber = make_fiber([section(0., 2.)], elements=points)
            return jnp.real(fiber.elements[0].scattering_matrix()[0, 0])

        reflectance = jnp.asarray(.04, dtype=jnp.float64)
        value = jax.jit(local_reflected_field)(reflectance)
        first = jax.jit(jax.grad(local_reflected_field))(reflectance)
        second = jax.jit(jax.grad(jax.grad(local_reflected_field)))(reflectance)
        cosine = math.cos(.3)
        np.testing.assert_allclose(value, .2*cosine, rtol=1e-14, atol=0.)
        np.testing.assert_allclose(first, cosine/.4, rtol=1e-13, atol=0.)
        np.testing.assert_allclose(second, -cosine/.032, rtol=1e-13, atol=0.)
    finally:
        jax.config.update("jax_enable_x64", previous)


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

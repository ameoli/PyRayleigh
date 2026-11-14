import array_api_compat.numpy as numpy_api
import array_api_strict as strict
import numpy as np
import pytest

from pyrayleigh import (Fiber, GuideOptics, PointReflector, RayleighRealization,
                        Section, WeakFBG, point_reflection_paths)

REFERENCE = 193.4e12


def make_fiber(elements=(), sample=None):
    guide = GuideOptics(REFERENCE, 1.4682, 1.4682)
    return Fiber("spool", 30., 293.15, REFERENCE - 50e9, REFERENCE + 50e9,
                  [Section(0., 30., guide, 0., 0.)], elements, sample)


def point(position, reflection=.2):
    return PointReflector(position, reflection, -reflection, .8, .8)


def test_two_reflectors_give_direct_and_repeated_cavity_returns():
    fiber = make_fiber([point(20.), point(5.)])
    assert point_reflection_paths(fiber, max_order=1) == ((0,), (1,))
    assert point_reflection_paths(fiber) == ((0,), (1,), (1, 0, 1))
    paths = point_reflection_paths(fiber, max_order=3)
    assert paths == ((0,), (1,), (1, 0, 1), (1, 0, 1, 0, 1))
    positions = [element.position for element in fiber.elements]
    apparent = [sum(positions[index] if event % 2 == 0 else -positions[index]
                    for event, index in enumerate(path)) for path in paths]
    assert apparent == [5., 20., 35., 50.]
    for path, distance in zip(paths, apparent):
        itinerary = [0., *(positions[index] for index in path), 0.]
        assert sum(abs(b - a) for a, b in zip(itinerary[:-1], itinerary[1:])) == 2.*distance


def test_three_reflectors_include_paths_between_different_cavities():
    fiber = make_fiber([point(3.), point(11.), point(25.)])
    paths = point_reflection_paths(fiber)
    assert paths == ((0,), (1,), (2,), (1, 0, 1), (1, 0, 2),
                     (2, 0, 1), (2, 0, 2), (2, 1, 2))


@pytest.mark.parametrize("count,expected", [(2, (2, 1, 1)), (3, (3, 5, 13)), (4, (4, 14, 70))])
def test_all_paths_are_unique_and_alternate_direction(count, expected):
    fiber = make_fiber([point(float(index)) for index in range(count)])
    paths = point_reflection_paths(fiber, max_order=3)
    assert isinstance(paths, tuple) and len(paths) == len(set(paths))
    assert tuple(sum(len(path) == 2*order - 1 for path in paths)
                 for order in (1, 2, 3)) == expected
    for path in paths:
        positions = [fiber.elements[index].position for index in path]
        for event, (left, right) in enumerate(zip(positions[:-1], positions[1:])):
            assert right < left if event % 2 == 0 else right > left


def test_only_points_reflect_and_indices_refer_to_the_full_element_tuple():
    elements = [WeakFBG(0., 1550e-9, .005, .01), point(0.),
                WeakFBG(5., 1550e-9, .005, .01), point(20.),
                WeakFBG(20., 1550e-9, .005, .01)]
    sample = RayleighRealization([.01j], [15.], [0., 30.])
    fiber = make_fiber(elements, sample)
    assert point_reflection_paths(fiber, max_order=3) == ((0,), (3,), (3, 0, 3), (3, 0, 3, 0, 3))
    assert fiber.rayleigh is sample
    assert all(actual is expected for actual, expected in zip(fiber.elements,
                [elements[1], elements[0], *elements[2:]]))


@pytest.mark.parametrize("elements", [[], [point(30.)], [WeakFBG(15., 1550e-9, .005, .01)]])
def test_fewer_than_two_points_have_no_multiple_returns(elements):
    fiber = make_fiber(elements)
    expected = tuple((index,) for index, element in enumerate(fiber.elements)
                     if isinstance(element, PointReflector))
    assert point_reflection_paths(fiber, max_order=3) == expected


@pytest.mark.parametrize("xp", [numpy_api, strict])
def test_zero_native_coefficients_do_not_change_the_geometry(xp):
    fiber = make_fiber([point(5., xp.asarray(0.)), point(20., xp.asarray(.3))])
    assert point_reflection_paths(fiber, max_order=np.int64(3)) == (
        (0,), (1,), (1, 0, 1), (1, 0, 1, 0, 1))


@pytest.mark.parametrize("order,error", [(True, TypeError), (np.bool_(True), TypeError),
                                         (2., TypeError), (np.array(2), TypeError),
                                         (0, ValueError), (-1, ValueError), (4, ValueError)])
def test_order_is_fixed_and_matches_the_supported_pyofdr_orders(order, error):
    with pytest.raises(error, match="max_order"):
        point_reflection_paths(make_fiber(), max_order=order)


def test_paths_require_a_fiber():
    with pytest.raises(TypeError, match="Fiber"):
        point_reflection_paths([point(5.), point(20.)])


def test_native_jax_coefficients_do_not_prune_paths_at_zero():
    jax = pytest.importorskip("jax")
    previous = jax.config.x64_enabled
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    try:
        def event_sum(reflection):
            fiber = make_fiber([point(5., reflection[0]), point(20., reflection[1])])
            total = jnp.asarray(0.)
            for path in point_reflection_paths(fiber):
                coefficient = jnp.asarray(1.)
                for event, index in enumerate(path):
                    element = fiber.elements[index]
                    coefficient = coefficient*(element.r_left if event % 2 == 0 else element.r_right)
                total = total + coefficient
            return total
        reflection = jnp.asarray([0., .3])
        np.testing.assert_allclose(jax.jit(event_sum)(reflection), .3)
        np.testing.assert_allclose(jax.jit(jax.grad(event_sum))(reflection), [.91, 1.])
        np.testing.assert_allclose(jax.jit(jax.hessian(event_sum))(reflection), [[0., -.6], [-.6, 0.]])
    finally:
        jax.config.update("jax_enable_x64", previous)

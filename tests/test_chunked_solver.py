"""Tests for the chunked normal-equation solver absorbed into
eigsep_sim.recovery from bloom21cm/src/linear_solver.py.

The core property worth pinning down is that chunking is an implementation
detail: the accumulated A^T A must match the dense result exactly regardless
of chunk size, and normal_solve_equations must agree with normal_solve.
"""

import numpy as np
import pytest

from eigsep_sim.recovery import (
    _build_A_chunk,
    build_A_right_product,
    build_normal_equations,
    normal_solve,
    normal_solve_equations,
)

N_OBS, N_ORBITS, NDIPOLE, NPIX = 5, 3, 2, 8
N_TOTAL = N_OBS * N_ORBITS


@pytest.fixture
def arrays():
    rng = np.random.default_rng(0)
    masks = rng.uniform(0, 1, size=(N_TOTAL, NPIX)).round()
    beams = rng.uniform(0.5, 1.5, size=(N_TOTAL, NDIPOLE, NPIX))
    omega_B = beams.sum(axis=2)
    j_sun = rng.integers(0, NPIX, size=N_OBS)
    return masks, beams, omega_B, j_sun


def _dense_A(masks, beams, omega_B, j_sun):
    return _build_A_chunk(
        masks, beams, omega_B, j_sun, NPIX, 0, N_TOTAL, N_OBS
    )


class TestChunkInvariance:
    @pytest.mark.parametrize("chunk", [1, 2, 4, 7, N_TOTAL, N_TOTAL + 10])
    def test_AtA_independent_of_chunk_size(self, arrays, chunk):
        masks, beams, omega_B, j_sun = arrays
        A = _dense_A(masks, beams, omega_B, j_sun)
        AtA, col_sums, _ = build_normal_equations(
            masks, beams, omega_B, j_sun, NPIX, chunk_obs=chunk
        )
        np.testing.assert_allclose(AtA, A.T @ A, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(
            col_sums, A.sum(axis=0), rtol=1e-12, atol=1e-12
        )

    def test_y_matches_dense_product(self, arrays):
        masks, beams, omega_B, j_sun = arrays
        A = _dense_A(masks, beams, omega_B, j_sun)
        x = np.arange(NPIX + 2, dtype=float)
        _, _, y = build_normal_equations(
            masks, beams, omega_B, j_sun, NPIX, x_true=x, chunk_obs=4
        )
        np.testing.assert_allclose(y, A @ x, rtol=1e-12, atol=1e-12)

    def test_y_is_none_without_x_true(self, arrays):
        masks, beams, omega_B, j_sun = arrays
        _, _, y = build_normal_equations(
            masks, beams, omega_B, j_sun, NPIX
        )
        assert y is None

    def test_column_layout(self, arrays):
        masks, beams, omega_B, j_sun = arrays
        A = _dense_A(masks, beams, omega_B, j_sun)
        assert A.shape == (N_TOTAL * NDIPOLE, NPIX + 2)
        # sky columns are the masked, normalized beam weights
        w = beams / omega_B[:, :, None]
        expected_sky = (w * masks[:, None, :]).reshape(
            N_TOTAL * NDIPOLE, NPIX
        )
        np.testing.assert_allclose(A[:, :NPIX], expected_sky)
        # regolith column is the beam weight over blocked pixels
        expected_reg = np.sum(
            w * (1.0 - masks[:, None, :]), axis=2
        ).reshape(N_TOTAL * NDIPOLE)
        np.testing.assert_allclose(A[:, NPIX], expected_reg)


class TestRightProduct:
    def test_AV_matches_dense(self, arrays):
        masks, beams, omega_B, j_sun = arrays
        A = _dense_A(masks, beams, omega_B, j_sun)
        rng = np.random.default_rng(1)
        V = rng.normal(size=(NPIX + 2, 4))
        AV, y, placeholder, At_Ax, col_sums = build_A_right_product(
            masks, beams, omega_B, j_sun, NPIX, V, chunk_obs=3
        )
        np.testing.assert_allclose(AV, (A @ V).astype(np.float32), rtol=1e-6)
        assert y is None and At_Ax is None and col_sums is None
        assert placeholder is None

    def test_optional_x_vec_products(self, arrays):
        masks, beams, omega_B, j_sun = arrays
        A = _dense_A(masks, beams, omega_B, j_sun)
        rng = np.random.default_rng(2)
        V = rng.normal(size=(NPIX + 2, 2))
        x = rng.normal(size=NPIX + 2)
        _, y, _, At_Ax, col_sums = build_A_right_product(
            masks, beams, omega_B, j_sun, NPIX, V, x_vec=x, chunk_obs=2
        )
        np.testing.assert_allclose(y, A @ x, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(
            At_Ax, A.T @ (A @ x), rtol=1e-10, atol=1e-10
        )
        np.testing.assert_allclose(col_sums, A.sum(axis=0), atol=1e-12)

    def test_output_is_float32(self, arrays):
        masks, beams, omega_B, j_sun = arrays
        V = np.ones((NPIX + 2, 2))
        AV, *_ = build_A_right_product(
            masks, beams, omega_B, j_sun, NPIX, V
        )
        assert AV.dtype == np.float32


class TestNormalSolveEquations:
    def test_agrees_with_dense_normal_solve(self, arrays):
        masks, beams, omega_B, j_sun = arrays
        # use a fully-observed mask so nothing is NaN
        masks = np.ones_like(masks)
        A = _dense_A(masks, beams, omega_B, j_sun)
        rng = np.random.default_rng(3)
        x_true = rng.uniform(100.0, 200.0, size=NPIX + 2)
        y = A @ x_true

        dense = normal_solve(A, y, NPIX)
        AtA, _, _ = build_normal_equations(
            masks, beams, omega_B, j_sun, NPIX, chunk_obs=2
        )
        chunked = normal_solve_equations(AtA, A.T @ y, NPIX)
        np.testing.assert_allclose(
            chunked["sky_map"], dense["sky_map"], rtol=1e-6, atol=1e-6
        )

    def test_flags_unobserved_pixels(self, arrays):
        masks, beams, omega_B, j_sun = arrays
        masks = np.ones_like(masks)
        masks[:, 2] = 0.0  # pixel 2 never visible
        A = _dense_A(masks, beams, omega_B, j_sun)
        AtA, _, _ = build_normal_equations(
            masks, beams, omega_B, j_sun, NPIX
        )
        out = normal_solve_equations(AtA, A.T @ np.ones(A.shape[0]), NPIX)
        assert out["unobserved"][2]
        assert np.isnan(out["sky_map"][2])

    def test_returns_expected_keys(self, arrays):
        masks, beams, omega_B, j_sun = arrays
        AtA, _, _ = build_normal_equations(
            masks, beams, omega_B, j_sun, NPIX
        )
        out = normal_solve_equations(AtA, np.ones(NPIX + 2), NPIX)
        assert set(out) == {
            "sky_map",
            "eigenvectors",
            "inv_eigenvalues",
            "unobserved",
        }

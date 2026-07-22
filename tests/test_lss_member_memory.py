"""
test_lss_member_memory.py
-------------------------
Memory-scaling guard for the ``--lss_marginalize`` LSS-completion member path.

The historical member path stored the FULL ``(M, N_rows, N_grid)`` missing-density
cube in the (per-proposal, traced, reverse-differentiated) prior state -- at the
builder default M=32, 49,152 rows, 1,000 grid nodes that is ONE float64 cube of
12.58 GB, before the reverse-mode intermediates and any K>1 multiply, which made
wide-sky ``--lss_marginalize`` non-runnable on common accelerators.

The refactor stores instead the member-INDEPENDENT ``base_miss`` curve ``(N_rows,
N_grid)`` and the compact per-member missing-mass table ``(M, N_rows)``, and
reconstructs each member's density two grid nodes at a time at the query
redshifts (the log-Q table stays a resident DATA constant on the catalog, gathered
but never densified into a derived cube).  So the RESIDENT prior-state bytes scale
like ``O(N_rows x N_grid) + O(M x N_rows)`` -- independent of the ``M x N_grid``
product that dominated the cube.

This test builds the marginalized state at a CPU-safe scale, prints the actual
state-pytree ``nbytes`` next to the equivalent old cube, and asserts the state is
FAR below the cube; it also value+grad-checks a tiny ``lss_marginalize`` likelihood
so the on-the-fly reconstruction is exercised end to end.  Run ``__main__`` on a
GPU for the full M=32 / 49,152-row scale (see the module docstring of the PR).
"""
import os

import numpy as np
import pytest

pytest.importorskip("jax")
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from darksirens.redshift import zgrid
from darksirens.core.types import CosmoParams, SurveyParams, EMCatalog, GWEvent
from jax.scipy.special import logsumexp

from darksirens.redshift.prior import (
    prepare_redshift_prior_state,
    eval_redshift_prior_members_with_state,
    DarkSirenEnsemblePriorState,
)
from darksirens.gw.populations import get_fixed_population_params

NG = int(zgrid.size)
_Z = np.asarray(zgrid)
COSMO = CosmoParams(H0=67.74, Om0=0.3075)
SURVEY = SurveyParams(n0=1.0, z50=0.15, w=0.08, delta=0.0, b_miss=0.0, alpha_miss=1.0)
POP = jnp.asarray(get_fixed_population_params("powerlaw+peak"))


def _state_nbytes(state):
    """Total device bytes of a prior-state pytree (nested leaves, scalars safe)."""
    total = 0
    for leaf in jax.tree_util.tree_leaves(state):
        arr = np.asarray(leaf)
        total += int(arr.nbytes)
    return total


def _marginalize_catalog(M, n_rows, *, seed=0):
    """Synthetic dark-siren catalog carrying an M-member LSS-completion ensemble.

    Values are arbitrary (only the SHAPES drive the memory footprint): a handful
    of galaxies per row, a directly-synthesised smoothed observed-density cache,
    a deterministic + an ensemble log-Q table.  ``unique_pixels=None`` so rows are
    already compact (the hot jit path)."""
    rng = np.random.default_rng(seed)
    nmax = 4
    zg = np.full((n_rows, nmax), 100.0)
    dz = np.full((n_rows, nmax), 1.0)
    w = np.zeros((n_rows, nmax))
    ng = np.zeros(n_rows, dtype=np.int32)
    # Two real galaxies per row at low z (arbitrary), padded otherwise.
    zg[:, 0] = 0.10
    zg[:, 1] = 0.14
    dz[:, :2] = 0.03
    w[:, :2] = 1.0
    ng[:] = 2
    # Synthesise the observed-density cache directly (skip the O(n_rows) KDE build);
    # a smooth positive bump keeps the completeness ratio well-defined.
    kde = np.exp(-0.5 * ((_Z[None, :] - 0.12) / 0.05) ** 2) * np.ones((n_rows, 1))
    logq = 0.2 * np.sin(np.linspace(0.0, 3.0, NG))[None, :] * np.ones((n_rows, 1))
    logq_members = (
        rng.standard_normal((M, 1, 1)) * (_Z[None, None, :] - 0.2)
    ) * np.ones((M, n_rows, NG))
    return EMCatalog(
        apix=1.0,
        zgals=jnp.asarray(zg), dzgals=jnp.asarray(dz), wgals=jnp.asarray(w),
        ngals=jnp.asarray(ng),
        delta_g_pix_z=jnp.zeros((1, NG)),
        dN_obs_kde=jnp.asarray(kde),
        pixel_to_cache_idx=jnp.arange(n_rows, dtype=jnp.int32),
        unique_pixels=None,
        lss_completion_logq=jnp.asarray(logq),
        lss_completion_logq_members=jnp.asarray(logq_members),
    )


def build_and_measure(M, n_rows, *, n_query=256, verbose=True):
    """Build the marginalized state at (M, n_rows), report state vs cube bytes, and
    value+grad a member-marginalized log-prior over a (z, pix) batch.

    The value+grad target is the LSS-member-marginalized redshift log-prior
    ``sum_query logsumexp_m log p_m(z | pix) - log M``: it exercises exactly the
    cube-free member path -- ``base_miss`` gathered at the query brackets, each
    member's Q reconstructed on the fly, and the streamed ``(M, N_rows)``
    normalizer -- and its gradient w.r.t. ``n0`` flows through ``base_miss`` (the
    completeness ``C`` and ``dN_exp``) and through the rematerialised member
    integral.  The FULL end-to-end ``darksiren_log_likelihood`` marginalized path
    is covered (finite value + logmeanexp identity + factored/reference value+grad
    parity) by tests/test_lss_marginalization.py and
    tests/test_member_factoring_parity.py.  Returns (state_nbytes, cube_nbytes)."""
    cat = _marginalize_catalog(M, n_rows)
    state = prepare_redshift_prior_state("dark_sirens", COSMO, SURVEY, cat)
    assert isinstance(state, DarkSirenEnsemblePriorState)

    state_nbytes = _state_nbytes(state)
    cube_nbytes = M * n_rows * NG * 8  # the OLD (M, N_rows, N_grid) float64 cube

    rng = np.random.default_rng(0)
    z_q = jnp.asarray(rng.uniform(0.05, 0.30, n_query))
    pix_q = jnp.asarray(rng.integers(0, n_rows, n_query), dtype=jnp.int32)

    def _logmix(n0):
        surv = SURVEY._replace(n0=n0)
        # materialize_state=False drops the (non-differentiable) optimization
        # barrier so jax.grad can cross the prior-state construction.
        st = prepare_redshift_prior_state(
            "dark_sirens", COSMO, surv, cat, materialize_state=False
        )
        lpm = eval_redshift_prior_members_with_state(
            "dark_sirens", st, z_q, pix_q, COSMO, surv, cat
        )  # (M, n_query)
        return jnp.sum(logsumexp(lpm, axis=0) - jnp.log(M))

    val = float(_logmix(SURVEY.n0))
    grad = float(jax.grad(_logmix)(SURVEY.n0))

    if verbose:
        print(
            f"\n[lss-member-memory] M={M} n_rows={n_rows} N_grid={NG}\n"
            f"  resident STATE pytree : {state_nbytes/1e9:8.4f} GB\n"
            f"  old (M,N_rows,N_grid) : {cube_nbytes/1e9:8.4f} GB (cube)\n"
            f"  state / cube          : {state_nbytes/cube_nbytes:8.4%}\n"
            f"  member log-prior sum  : {val:.6f}\n"
            f"  d(sum)/d n0           : {grad:.6f}"
        )
    assert np.isfinite(val), "marginalized member log-prior not finite"
    assert np.isfinite(grad), "marginalized member log-prior gradient not finite"
    return state_nbytes, cube_nbytes


def test_state_scales_below_cube():
    """At a CPU-safe scale the resident state is FAR below the old member cube."""
    M, n_rows = 8, 4096
    state_nbytes, cube_nbytes = build_and_measure(M, n_rows)
    # base_miss + dN_miss (2 * n_rows * NG) + log_Z_members (M * n_rows) dominate;
    # the cube is M * n_rows * NG.  With M=8 the state should be < 40% of the cube
    # (the ratio -> 2/M as M grows: at M=32 it is ~6%).
    assert state_nbytes < 0.4 * cube_nbytes, (
        f"state {state_nbytes/1e9:.3f} GB not far below cube "
        f"{cube_nbytes/1e9:.3f} GB"
    )
    # And the state must not itself contain an (M, N_rows, N_grid) leaf.
    assert state_nbytes < 1.5 * (2 * n_rows * NG + M * n_rows) * 8


if __name__ == "__main__":  # pragma: no cover -- GPU full-scale benchmark
    M = int(os.environ.get("LSS_BENCH_M", "32"))
    n_rows = int(os.environ.get("LSS_BENCH_ROWS", "49152"))
    build_and_measure(M, n_rows)
    print(f"[lss-member-memory] backend: {jax.devices()[0].platform}")

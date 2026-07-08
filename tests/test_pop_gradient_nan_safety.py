"""NUTS safety: log_p_pop hyperparameter gradients must stay finite when the
population density is exactly zero for some samples (below the low-mass
taper, beyond support, or underflowed products like the sigma_chi Gaussian
far in its tails).

Pre-fix, six ``jnp.where(p > 0.0, jnp.log(p), -jnp.inf)`` sites produced
``0 * (1/0) = NaN`` cotangents in the backward pass whenever any sample had
``p == 0``; the NaN poisons the summed hyperparameter gradient, and every
NumPyro NUTS trajectory that touches such a point is flagged divergent (the
H1 smoke measured 100% divergent transitions from exactly this). The fix is
the same ``jnp.log(jnp.maximum(p, 1e-300))`` idiom the redshift module
already used.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from darksirens.gw.populations import pop_model_parser
from darksirens.gw.populations.registry import get_fixed_population_params

MODELS = [
    "powerlaw+peak",
    "brokenpowerlaw+2peaks",
    "gwtc5_fiducial_bpl2peaks",
    "gp1d_m1",
]

# Some parametric test modules insert a tinygp STUB into sys.modules at import
# time; GP models then cannot evaluate. Follow the registry-golden convention:
# a stub (no __file__) counts as "tinygp unavailable".
try:
    import tinygp  # noqa: F401
    HAVE_TINYGP = getattr(tinygp, "__file__", None) is not None
except Exception:  # pragma: no cover
    HAVE_TINYGP = False


@pytest.mark.parametrize("pop_model", MODELS)
def test_log_p_pop_gradient_finite_at_zero_density(pop_model):
    if pop_model == "gp1d_m1" and not HAVE_TINYGP:
        pytest.skip("tinygp unavailable (or stubbed by a parametric test module)")
    log_p_pop = pop_model_parser(pop_model)
    theta = jnp.asarray(get_fixed_population_params(pop_model), dtype=float)

    # Samples straddling the support: inside the bulk, below any plausible
    # low-mass cutoff (density exactly 0 through the taper's hard edge), far
    # above the high-mass end, and with an extreme chieff so a narrow spin
    # Gaussian underflows to exactly 0.
    m1 = jnp.asarray([35.0, 1.2, 500.0, 35.0])
    q = jnp.asarray([0.8, 0.9, 0.5, 0.8])
    z = jnp.asarray([0.3, 0.3, 0.3, 0.3])
    chieff = jnp.asarray([0.0, 0.0, 0.0, 0.99])

    def total(th):
        lp = log_p_pop(m1, q, z, chieff, th)
        lp = jnp.where(jnp.isfinite(lp), lp, -jnp.inf)
        # mirror the likelihood's per-event logsumexp reduction over samples
        return jax.scipy.special.logsumexp(lp)

    val = float(total(theta))
    grad = np.asarray(jax.grad(total)(theta))
    assert np.isfinite(val)
    assert np.all(np.isfinite(grad)), (
        f"{pop_model}: non-finite hyperparameter gradient "
        f"{dict(enumerate(grad.tolist()))}"
    )

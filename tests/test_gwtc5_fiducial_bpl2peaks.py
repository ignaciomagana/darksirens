import jax.numpy as jnp
import numpy as np

from darksirens.gw.populations.registry import (
    get_fixed_population_params,
    get_model,
    pop_model_prior_parser,
)


MODEL = "gwtc5_fiducial_brokenpowerlaw+2peaks"


def test_gwtc5_fiducial_bpl2peaks_table5_priors_and_fiducial_order():
    lower, upper, labels, latex = pop_model_prior_parser(MODEL)

    expected = [
        (r"$\alpha_1$", -4.0, 12.0),
        (r"$\alpha_2$", -4.0, 12.0),
        (r"$m_{\rm break}$", 20.0, 50.0),
        (r"$\mu_1$", 5.0, 20.0),
        (r"$\sigma_1$", 0.0, 10.0),
        (r"$\mu_2$", 25.0, 60.0),
        (r"$\sigma_2$", 0.0, 10.0),
        (r"$m_{1,{\rm low}}$", 3.0, 10.0),
        (r"$\delta m_1$", 0.0, 10.0),
        (r"$\lambda_0$", 0.0, 1.0),
        (r"$\lambda_1$", 0.0, 1.0),
        (r"$\beta_q$", -2.0, 7.0),
        (r"$m_{2,{\rm low}}$", 3.0, 10.0),
        (r"$\delta m_2$", 0.0, 10.0),
    ]

    assert list(zip(labels[:14], lower[:14], upper[:14])) == expected
    assert latex == r"\text{GWTC-5 Fiducial BPL+2G}"
    assert len(get_fixed_population_params("gwtc5_brokenpowerlaw+2peaks")) == len(labels)
    assert len(get_fixed_population_params("gwtc5_fiducial_bpl2peaks")) == len(labels)

    theta = get_fixed_population_params(MODEL)
    assert len(theta) == len(labels)
    assert labels[8] == r"$\delta m_1$"
    assert labels[13] == r"$\delta m_2$"
    assert theta[8] == 3.0
    assert theta[13] == 3.0


def test_gwtc5_fiducial_bpl2peaks_log_population_is_finite_and_enforces_constraints():
    model = get_model(MODEL)
    theta = get_fixed_population_params(MODEL)

    logp = model.log_p_pop(
        jnp.array([20.0, 40.0]),
        jnp.array([0.8, 0.9]),
        jnp.array([0.1, 0.2]),
        jnp.array([0.0, 0.05]),
        theta,
    )
    assert np.all(np.isfinite(np.asarray(logp)))

    # Table 5 has m2_low ~ U(3, m1_low) and lambda_0, lambda_1 on a simplex.
    bad_m2_low = theta.at[12].set(theta[7] + 0.5)
    assert np.all(np.isneginf(np.asarray(model.log_p_pop(20.0, 0.8, 0.1, 0.0, bad_m2_low))))

    bad_simplex = theta.at[9].set(0.8).at[10].set(0.4)
    assert np.all(np.isneginf(np.asarray(model.log_p_pop(20.0, 0.8, 0.1, 0.0, bad_simplex))))

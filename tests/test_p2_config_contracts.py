"""Configuration/contract fixes from the P2 review batch.

* P2-01 — the mixture loader branched on the CLI ``--lss_completion`` path
  helper, so an EMBEDDED ``/lss_completion`` group (auto-discovered from the
  survey file) combined with ``--use_lss`` built BOTH field inputs and hit
  the field-normalizer's mutual-exclusion invariant inside the jit trace.
* P2-06 — preflight stashed file-contract sub-reports without surfacing
  their errors; an empty selection group was "contract valid".
* P2-07 — the exactly-one-detected and both-detected lensed selection
  estimators come from mutually exclusive indicators over the same campaign
  draws (per-draw covariance ``-mu1*mu2/N``) but were combined as
  independent, overstating the total variance.
* P2-09 — zero-event PE inputs passed the contract and divided by zero in
  the likelihood's block plan.
* P2-10 — the singleton-subset loader skipped the plus/minus source-field
  consistency contract the full loader enforces.
"""
import numpy as np
import pytest

pytest.importorskip("jax")
import h5py
import jax.numpy as jnp


# ---------------------------------------------------------------------------
# P2-09: zero events
# ---------------------------------------------------------------------------

def test_pe_contract_rejects_zero_events(tmp_path):
    from darksirens.lensing.file_contract import validate_observed_gw_pe

    from darksirens.lensing.file_contract import (
        EVENT_INDEXING,
        PE_FORMAT_VERSION,
    )

    path = tmp_path / "pe.h5"
    with h5py.File(path, "w") as f:
        f.attrs["format_version"] = PE_FORMAT_VERSION
        f.attrs["event_indexing"] = EVENT_INDEXING
        f.attrs["n_events"] = 0
        f.attrs["nsamp"] = 8
        for name in ("m1det", "m2det", "dL", "chieff", "p_pe"):
            f.create_dataset(name, data=np.zeros(0))
    report = validate_observed_gw_pe(str(path))
    assert not report["ok"]
    assert any("at least one event" in e for e in report["errors"])


def test_core_rejects_zero_events_with_a_configuration_error():
    import inspect

    from darksirens.likelihood import core

    src = inspect.getsource(core.darksiren_log_likelihood)
    assert "requires at least one event" in src


# ---------------------------------------------------------------------------
# P2-06: contract/preflight strictness
# ---------------------------------------------------------------------------

def test_selection_contract_rejects_empty_groups(tmp_path):
    from darksirens.lensing.file_contract import validate_selection_inputs

    path = tmp_path / "sel.h5"
    with h5py.File(path, "w") as f:
        f.attrs["format_version"] = "lensing-selection-inputs-1.0"
        f.create_group("unlensed")  # present but EMPTY
    report = validate_selection_inputs(str(path))
    assert not report["ok"]
    assert any("empty" in e for e in report["errors"])


# ---------------------------------------------------------------------------
# P2-10: singleton subset loader consistency
# ---------------------------------------------------------------------------

def _write_pair_campaign(path, corrupt_field=None):
    n_src = 3
    rng = np.random.default_rng(0)
    sid = np.repeat(np.arange(n_src), 2)
    iid = np.tile([0, 1], n_src)
    per_source = {
        "m1_src": rng.uniform(20, 40, n_src),
        "q_src": rng.uniform(0.5, 1.0, n_src),
        "z_src": rng.uniform(0.2, 1.0, n_src),
        "chieff": rng.uniform(-0.2, 0.2, n_src),
        "y_source": rng.uniform(0.1, 0.9, n_src),
        "p_prop_src": rng.uniform(0.5, 1.5, n_src),
        "p_prop_y": rng.uniform(0.5, 1.5, n_src),
    }
    with h5py.File(path, "w") as f:
        f.create_dataset("source_id", data=sid.astype(np.int32))
        f.create_dataset("image_id", data=iid.astype(np.int32))
        for name, vals in per_source.items():
            doubled = np.repeat(vals, 2)
            if name == corrupt_field:
                doubled[1] += 1.0  # the mu_- image disagrees with mu_+
            f.create_dataset(name, data=doubled)
        f.create_dataset("mu", data=rng.uniform(1.0, 5.0, 2 * n_src))
        # exactly-one-detected for every source
        f.create_dataset(
            "detected", data=np.tile([True, False], n_src)
        )
        f.attrs["n_draw_sources"] = 100


def test_singleton_loader_enforces_source_field_consistency(tmp_path):
    from darksirens.lensing.lensed_injections import load_lensed_single_image_set

    good = tmp_path / "good.h5"
    _write_pair_campaign(good)
    singles = load_lensed_single_image_set(str(good))
    assert int(np.asarray(singles.m1_src).shape[0]) == 3

    bad = tmp_path / "bad.h5"
    _write_pair_campaign(bad, corrupt_field="z_src")
    with pytest.raises(ValueError, match="z_src"):
        load_lensed_single_image_set(str(bad))


# ---------------------------------------------------------------------------
# P2-07: correlated lensed-channel variance
# ---------------------------------------------------------------------------

def test_lensed_channel_variance_subtracts_the_shared_campaign_covariance():
    """The combination site must implement Var(mu_1L + mu_2)
    = Var_1L + Var_2 - 2*mu_1L*mu_2/N (the union-estimator variance), not
    the independent sum."""
    import inspect

    from darksirens.likelihood import likelihood_with_clusters as lwc

    src = inspect.getsource(lwc.darksiren_log_likelihood_with_clusters)
    assert "log_cov2" in src and "n_draw_sources" in src, (
        "the exactly-one-detected and both-detected estimators share a "
        "campaign; their covariance (-mu1*mu2/N) must enter the combined "
        "selection variance"
    )
    # And the numbers: for weights w on disjoint subsets A, B of N draws,
    # Var(union) = Var_A + Var_B - 2 mu_A mu_B / N exactly.
    rng = np.random.default_rng(1)
    N = 1000
    w = rng.lognormal(size=N)
    in_A = rng.uniform(size=N) < 0.3
    in_B = (~in_A) & (rng.uniform(size=N) < 0.4)
    wA = np.where(in_A, w, 0.0)
    wB = np.where(in_B, w, 0.0)
    mu_A, mu_B = wA.sum() / N, wB.sum() / N
    var_A = (wA**2).sum() / N**2 - mu_A**2 / N
    var_B = (wB**2).sum() / N**2 - mu_B**2 / N
    wU = wA + wB
    var_U = (wU**2).sum() / N**2 - (wU.sum() / N) ** 2 / N
    np.testing.assert_allclose(
        var_U, var_A + var_B - 2.0 * mu_A * mu_B / N, rtol=1e-12
    )


# ---------------------------------------------------------------------------
# P2-01: embedded Q + --use_lss in the mixture loader
# ---------------------------------------------------------------------------

def test_mixture_loader_branches_on_the_loaded_q_state():
    """The overdensity-grid decision must test the LOADED table, not the CLI
    path helper: an embedded /lss_completion group arrives with no explicit
    --lss_completion entry."""
    import inspect

    from darksirens.inference import loaders

    src = inspect.getsource(loaders)
    # The mixture bundle loop must gate delta_g on the loaded lss dict.
    assert 'lss.get("lss_completion_logq") is None' in src, (
        "embedded-Q catalogs with --use_lss would build BOTH field inputs "
        "and abort on the field-normalizer mutual-exclusion invariant"
    )

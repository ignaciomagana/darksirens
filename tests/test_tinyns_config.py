from argparse import Namespace

import pytest

from darksirens.inference.tinyns_config import build_tinyns_config, parse_chain_schedule
from darksirens.cli.inference import main  # import smoke


def ns(**kw):
    base = dict(
        sampler="tinyns", tinyns_preset="recommended", nlive=100, dlogz=0.1,
        max_samples=1000, seed=1, show_progress=False,
    )
    names = [
        "sample","kernel","vectorized","max_attempts","walks","step_scale",
        "batch_size","min_accepts","replacement_chains","replacement_chain_schedule",
        "rwalk_proposal","rwalk_cov_jitter","bound","bound_enlargement",
        "bound_update_interval","bound_jitter","bound_max_draws",
        "multi_bound_max_ellipsoids","multi_bound_min_points",
        "multi_bound_split_threshold","multi_bound_enlargement",
        "multi_bound_overlap_correction","rwalk_seed","rwalk_seed_fallback",
        "bound_seed_kernel","allow_unused_bound","fused_bound_rwalk",
        "bound_rebuild_on_failure","bound_failure_rebuild_threshold",
        "jax_vectorized","jax_block_size","checkpoint_path","checkpoint_interval",
        "resume_from","checkpoint_path_out","progress_interval",
    ]
    base.update({f"tinyns_{n}": None for n in names})
    base.update(kw)
    return Namespace(**base)


def test_recommended_defaults():
    c = build_tinyns_config(ns())
    assert (c.sample, c.kernel, c.rwalk_proposal, c.walks) == ("rwalk", "jax", "isotropic", 5)
    assert c.replacement_chains == 1
    assert c.bound == "none"
    assert c.jax_block_size == 32


def test_python_debug_defaults():
    c = build_tinyns_config(ns(tinyns_preset="python_debug"))
    assert (c.sample, c.kernel, c.bound, c.jax_block_size) == ("rwalk", "python", "none", 1)


def test_prior_defaults():
    c = build_tinyns_config(ns(tinyns_preset="prior"))
    assert (c.sample, c.kernel, c.bound, c.replacement_chains) == ("prior", "python", "none", 1)


def test_explicit_override_beats_preset():
    assert build_tinyns_config(ns(tinyns_walks=11)).walks == 11


def test_slice_and_rslice_not_valid_samples():
    with pytest.raises(ValueError):
        build_tinyns_config(ns(tinyns_sample="slice"))
    with pytest.raises(ValueError):
        build_tinyns_config(ns(tinyns_sample="rslice"))


def test_schedule_parsing():
    assert parse_chain_schedule("1,4,16") == (1, 4, 16)
    assert parse_chain_schedule("(1,4,16)") == (1, 4, 16)


def test_non_increasing_schedule_fails():
    with pytest.raises(ValueError):
        build_tinyns_config(ns(tinyns_replacement_chain_schedule="1,16,4"))


def test_schedule_and_replacement_chains_fail():
    with pytest.raises(ValueError):
        build_tinyns_config(ns(tinyns_replacement_chain_schedule="1,4,16", tinyns_replacement_chains=2))


def test_replacement_chains_with_python_fails():
    with pytest.raises(ValueError):
        build_tinyns_config(ns(tinyns_kernel="python", tinyns_replacement_chains=2, tinyns_jax_block_size=1))


def test_jax_block_size_with_python_fails():
    with pytest.raises(ValueError):
        build_tinyns_config(ns(tinyns_kernel="python", tinyns_jax_block_size=2))


def test_jax_block_size_unbounded_rwalk_jax_passes():
    assert build_tinyns_config(ns(tinyns_jax_block_size=16)).jax_block_size == 16


def test_bound_multi_live_seed_requires_allow_unused_bound():
    with pytest.raises(ValueError):
        build_tinyns_config(ns(tinyns_bound="multi", tinyns_rwalk_seed="live", tinyns_jax_block_size=1))
    assert build_tinyns_config(ns(tinyns_bound="multi", tinyns_rwalk_seed="live", tinyns_allow_unused_bound=True, tinyns_jax_block_size=1)).allow_unused_bound


def test_fused_bound_none_fails():
    with pytest.raises(ValueError):
        build_tinyns_config(ns(tinyns_fused_bound_rwalk=True, tinyns_bound="none"))


def test_bounded_fused_block_combo_passes():
    c = build_tinyns_config(ns(tinyns_sample="rwalk", tinyns_kernel="jax", tinyns_bound="multi", tinyns_rwalk_seed="bound", tinyns_bound_seed_kernel="jax", tinyns_fused_bound_rwalk=True, tinyns_jax_block_size=16))
    assert c.jax_block_size == 16


def test_explicit_max_attempts_too_small_fails():
    with pytest.raises(ValueError):
        build_tinyns_config(ns(tinyns_walks=25, tinyns_replacement_chains=16, tinyns_max_attempts=100))


def test_unset_max_attempts_auto_resolves():
    c = build_tinyns_config(ns(tinyns_walks=25, tinyns_replacement_chain_schedule="1,4,16"))
    assert c.max_attempts >= 25 * 16


def test_checkpoint_resume_conflict():
    with pytest.raises(ValueError):
        build_tinyns_config(ns(tinyns_resume_from="in.chk", tinyns_checkpoint_path="same.chk", tinyns_checkpoint_interval=10))
    c = build_tinyns_config(ns(tinyns_resume_from="in.chk", tinyns_checkpoint_path="new.chk", tinyns_checkpoint_path_out="out.chk", tinyns_checkpoint_interval=10))
    assert c.checkpoint_path_out == "out.chk"

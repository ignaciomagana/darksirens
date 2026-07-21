"""Run-directory naming/collision handling (library review, BUG-5): the run
name used to be pop_model/universe_model/sampler/timestamp at one-second
resolution with os.makedirs(exist_ok=True), so two same-config jobs started
in the same second silently shared a directory and one clobbered the
other's results.hdf5."""
import os
from types import SimpleNamespace

from darksirens.cli.inference import _make_run_dir


def _opts(save_path, **overrides):
    base = dict(
        pop_model="powerlaw+peak", universe_model="dark_sirens", sampler="tinyns",
        seed=7,
    )
    base.update(overrides)
    return SimpleNamespace(save_path=str(save_path), **base)


def test_make_run_dir_includes_seed_in_name(tmp_path):
    run_dir = _make_run_dir(_opts(tmp_path), "2026-07-21T00-00-00")
    assert "seed7" in run_dir


def test_make_run_dir_disambiguates_same_second_collisions(tmp_path):
    """Two identically-configured jobs invoked with the same frozen
    timestamp must land in two distinct directories, not overwrite."""
    opts = _opts(tmp_path)
    timestamp = "2026-07-21T00-00-00"

    first  = _make_run_dir(opts, timestamp)
    second = _make_run_dir(opts, timestamp)

    assert first != second
    assert os.path.isdir(first)
    assert os.path.isdir(second)


def test_make_run_dir_retries_beyond_one_collision(tmp_path):
    opts = _opts(tmp_path)
    timestamp = "2026-07-21T00-00-00"

    made = [_make_run_dir(opts, timestamp) for _ in range(3)]

    assert len(set(made)) == 3
    assert made[1].endswith("-01")
    assert made[2].endswith("-02")

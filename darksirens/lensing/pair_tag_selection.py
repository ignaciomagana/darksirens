"""Deterministic mock pair-tag selection models for simulated lensing studies.

These models describe the probability that an already both-detected lensed
image pair is identified/tagged as a candidate pair.  They are simulation-only
scaffolding and are not calibrated to GWTC-5 or any real search pipeline.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Mapping, Any

import numpy as np


# The tuple itself lives in the dependency-free darksirens.core.constants so the
# lensing CLI's argparse choices do not have to import this package; this stays
# the canonical spelling.
from darksirens.core.constants import (  # noqa: E402,F401
    PAIR_TAG_SELECTION_MODEL_KINDS,
)


def _sigmoid(x):
    x = np.asarray(x, dtype=float)
    return 1.0 / (1.0 + np.exp(-np.clip(x, -700.0, 700.0)))


def _logit(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-12, 1.0 - 1e-12)
    return np.log(p) - np.log1p(-p)


def _broadcast_field_shape(fields) -> tuple:
    """Common shape of the supplied per-source fields, or ``()`` if none.

    Lets the constant/none kinds return the same per-source shape as the
    score-based kinds without every caller having to pass a length.
    """
    shapes = [np.shape(np.asarray(v, dtype=float)) for v in fields.values() if v is not None]
    if not shapes:
        return ()
    return np.broadcast_shapes(*shapes)


@dataclass(frozen=True)
class PairTagSelectionModel:
    """Simple deterministic pair-identification/tagging probability model."""

    kind: str = "constant"
    constant: float = 1.0
    perturb_logit: float = 0.0

    @property
    def required_fields(self) -> tuple[str, ...]:
        if self.kind in ("none", "constant"):
            return ()
        if self.kind == "snr_only":
            return ("snr_image0", "snr_image1")
        if self.kind == "snr_sky":
            return ("snr_image0", "snr_image1", "log_sky_overlap")
        if self.kind == "snr_time":
            return ("snr_image0", "snr_image1", "delta_t_obs")
        if self.kind == "snr_time_sky":
            return ("snr_image0", "snr_image1", "delta_t_obs", "log_sky_overlap")
        raise ValueError(f"unknown pair tag selection model: {self.kind}")

    def probability(self, **fields) -> np.ndarray:
        """Return p_tag in [0, 1] for the configured model.

        The result is always shaped like the supplied per-source fields, for
        EVERY kind. The constant kinds used to return a 0-d array while the
        score-based kinds returned a per-source array, so a caller that indexed
        the result — e.g. the mock generator's
        ``tagged_pair[both] = rng.uniform(...) < p_tag[both]`` — crashed with
        ``IndexError: invalid index to scalar variable`` under
        ``--pair-tag-model constant``. When no fields are supplied there is no
        length to broadcast to and the result stays 0-d (the inference side
        calls it that way and broadcasts the log-weight itself).
        """
        if self.kind in ("none", "constant"):
            p = np.full(_broadcast_field_shape(fields), float(self.constant), dtype=float)
        elif self.kind in ("snr_only", "snr_sky", "snr_time", "snr_time_sky"):
            snr0 = np.asarray(fields["snr_image0"], dtype=float)
            snr1 = np.asarray(fields["snr_image1"], dtype=float)
            min_snr = np.minimum(snr0, snr1)
            score = -1.25 + 0.32 * (min_snr - 8.0)
            if self.kind in ("snr_time", "snr_time_sky"):
                dt = np.abs(np.asarray(fields.get("delta_t_obs", fields.get("true_delta_t")), dtype=float))
                score = score + 0.10 * np.log1p(dt / 86400.0)
            if self.kind in ("snr_sky", "snr_time_sky"):
                score = score + 0.25 * np.asarray(fields["log_sky_overlap"], dtype=float)
            p = _sigmoid(score)
        else:
            raise ValueError(f"unknown pair tag selection model: {self.kind}")
        if self.perturb_logit:
            p = _sigmoid(_logit(p) + float(self.perturb_logit))
        return np.clip(np.asarray(p, dtype=float), 0.0, 1.0)

    def log_probability(self, **fields) -> np.ndarray:
        p = self.probability(**fields)
        with np.errstate(divide="ignore"):
            return np.where(p > 0.0, np.log(np.where(p > 0.0, p, 1.0)), -np.inf)


def make_pair_tag_selection_model(kind: str = "constant", *, constant: float = 1.0, perturb_logit: float = 0.0) -> PairTagSelectionModel:
    if kind == "file":
        raise ValueError("kind='file' requires load_pair_tag_selection_file")
    if not (0.0 <= float(constant) <= 1.0):
        raise ValueError("pair_tag_constant must be in [0, 1]")
    return PairTagSelectionModel(kind=kind, constant=float(constant), perturb_logit=float(perturb_logit))


def load_pair_tag_selection_file(path: str | Path, *, perturb_logit: float = 0.0) -> PairTagSelectionModel:
    with open(path, "r", encoding="utf-8") as f:
        data: Mapping[str, Any] = json.load(f)
    return make_pair_tag_selection_model(
        str(data.get("kind", data.get("model", "constant"))),
        constant=float(data.get("constant", 1.0)),
        perturb_logit=float(data.get("perturb_logit", 0.0)) + float(perturb_logit),
    )

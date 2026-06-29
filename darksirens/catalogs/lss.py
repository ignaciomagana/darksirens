"""LSS completion table loading and validation."""

import numpy as np


GALAXY_AWARE_MODELS = ["dark_sirens", "dark_sirens_complete"]


def maybe_load_lss_completion(opts, *, zgrid) -> dict:
    """Load optional LSS-conditioned lognormal completion tables."""
    lss_completion_logq = None
    lss_completion_logq_members = None
    lss_completion_indexing = 0  # int enum: 0=auto, 1=compact, 2=global
    lss_path = getattr(opts, "lss_completion", None)
    if lss_path is None and opts.survey_path is not None:
        try:
            import h5py
            with h5py.File(opts.survey_path, "r") as _f:
                if "lss_completion" in _f:
                    lss_path = opts.survey_path
        except Exception:
            lss_path = None
    if lss_path is not None and opts.universe_model in GALAXY_AWARE_MODELS:
        from darksirens.em.lognormal_completion import load_lss_completion_hdf5
        loaded = load_lss_completion_hdf5(lss_path)
        logq = loaded.get("logq_map")
        if logq is None:
            raise ValueError(
                f"LSS completion file '{lss_path}' has no /lss_completion/logq_map "
                "(deterministic table required for inference)."
            )
        logq = np.asarray(logq, dtype=float)
        if logq.shape[-1] != len(zgrid):
            raise ValueError(
                f"LSS completion N_grid={logq.shape[-1]} but the package zgrid has "
                f"size {len(zgrid)}; rebuild the completion on the package grid."
            )
        zg_file = loaded.get("zgrid")
        if zg_file is not None and not np.allclose(
            np.asarray(zg_file, dtype=float), np.asarray(zgrid, dtype=float),
            rtol=1e-5, atol=1e-8,
        ):
            raise ValueError(
                "LSS completion zgrid does not match the package zgrid (no silent interpolation)."
            )
        lss_completion_logq = np.asarray(logq)
        lss_completion_indexing = {"compact": 1, "global": 2}.get(
            str(loaded.get("indexing", "compact")), 0
        )
        if getattr(opts, "lss_marginalize", False):
            logq_m = loaded.get("logq_members")
            if logq_m is None:
                raise ValueError(
                    f"--lss_marginalize requires an LSS-completion ENSEMBLE, but "
                    f"'{lss_path}' has no /lss_completion/logq_members. Rebuild Q with "
                    "darksirens_build_lognormal_completion --n-members M (M > 0)."
                )
            logq_m = np.asarray(logq_m, dtype=float)
            if logq_m.shape[-1] != len(zgrid):
                raise ValueError(
                    f"LSS completion members N_grid={logq_m.shape[-1]} but the package "
                    f"zgrid has size {len(zgrid)}; rebuild on the package grid."
                )
            lss_completion_logq_members = logq_m
            print(
                f"    - LSS completion ENSEMBLE loaded: logq_members "
                f"{tuple(logq_m.shape)} (M={logq_m.shape[0]}) for fully-Bayesian "
                "marginalisation over the missing-galaxy field"
            )
        print(
            f"    - LSS completion loaded from {lss_path}: logq_map {tuple(logq.shape)}, "
            f"indexing={loaded.get('indexing')}"
        )
        _diag = loaded.get("diagnostics") or {}
        _fid = {k: _diag[k] for k in (
            "fiducial_H0", "fiducial_Om0", "fiducial_n0", "fiducial_delta",
            "bias_b_miss", "lss_corr_length_mpc", "lss_sigma",
        ) if k in _diag}
        if _fid:
            print(f"    - Q_LSS build fiducials: {_fid}")
        print(
            "    [!] Q_LSS is FIXED at its build-time fiducials (cosmology, n0, "
            "delta, bias); the inference will vary some of these. Q is a "
            "radial completion field on the SAME zgrid (validated), interpreted "
            "as a dimensionless density-ratio. Rebuild Q if your fiducials differ "
            "substantially."
        )
    return {
        "lss_completion_logq": lss_completion_logq,
        "lss_completion_logq_members": lss_completion_logq_members,
        "lss_completion_indexing": lss_completion_indexing,
    }

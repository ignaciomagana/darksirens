"""LSS completion table loading and validation."""

import numpy as np

from darksirens.core.model_kinds import GALAXY_AWARE_MODELS


def maybe_load_lss_completion(opts, *, zgrid) -> dict:
    """Load optional LSS-conditioned lognormal completion tables."""
    lss_completion_logq = None
    lss_completion_logq_members = None
    lss_completion_indexing = 0  # int enum: 0=auto, 1=compact, 2=global
    lss_completion_provenance = None  # ensemble-provenance for K>=2 verification
    lss_completion_fiducials = None   # build-time conditioning values (None = no Q)
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
        from darksirens.redshift.lognormal_completion import load_lss_completion_hdf5
        loaded = load_lss_completion_hdf5(lss_path)
        # c_mode provenance, fail-closed: Q is fit RESIDUAL to a completeness
        # base -- the per-pixel ratio (which absorbs the observed angular
        # clustering) or the sky-aggregate Cbar (which leaves the full
        # clustering to Q).  The two targets differ by the entire clustering
        # signal, so consuming one as the other double-counts or drops it.
        # An absent attr is a legacy table: always per-pixel-base.
        table_c_mode = loaded.get("c_mode") or "per_pixel"
        survey_c_mode = str(getattr(opts, "c_mode", None) or "per_pixel")
        if table_c_mode != survey_c_mode:
            raise ValueError(
                f"LSS completion c_mode mismatch: table '{lss_path}' was built "
                f"against the {table_c_mode!r} completeness base"
                + (" (no c_mode attr: legacy per-pixel-base table)"
                   if loaded.get("c_mode") is None else "")
                + f", but this run's survey uses c_mode={survey_c_mode!r}. Q is "
                "the fit residual to its completeness base, and the two bases "
                "differ by the entire observed clustering signal -- consuming a "
                "per-pixel-base table in aggregate mode double-counts the "
                "clustering, and the reverse drops it. Rebuild the table with "
                f"darksirens_build_lognormal_completion --c-mode "
                f"{survey_c_mode}, or run with --c_mode {table_c_mode}."
            )
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
        # Ensemble provenance travels with the loaded table so a K>=2 mixture
        # can verify the per-catalog ensembles share a matched realization set
        # before marginalizing over a shared member index (None values here mean
        # a legacy file with no provenance attrs).
        lss_completion_provenance = {
            "path": lss_path,
            "realization_set_id": loaded.get("realization_set_id"),
            "member_content_sha256": loaded.get("member_content_sha256"),
            "n_members": loaded.get("n_members"),
        }
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
        # NOTE: this whitelist is the INPUT to the provenance guard, so every
        # key named in ``q_provenance._Q_CONDITIONED`` must appear here.
        # fiducial_w0/fiducial_wa were previously missing, which silently made
        # the guard dead code for the two dark-energy parameters it claims to
        # protect (the builder stamps them; see build_lognormal_completion.py).
        _fid = {k: _diag[k] for k in (
            "fiducial_H0", "fiducial_Om0", "fiducial_w0", "fiducial_wa",
            "fiducial_n0", "fiducial_delta",
            "bias_b_miss", "lss_corr_length_mpc", "lss_sigma",
            # Parametric-selection base (c_mode="selection" tables): the
            # theta_hat the fixed base was built at, checked in the CLI
            # against the --selection_fit prior center.
            "selection_m_lim", "selection_M0hat", "selection_sigma_M",
        ) if k in _diag}
        # K(z) template coefficients of a selection base (variable count,
        # stamped per coefficient as selection_kcorr_c1, c2, ...); absent on
        # pre-K tables, meaning K = 0.  Compared elementwise in the CLI
        # against the --selection_fit template.
        _fid.update({k: _diag[k] for k in _diag
                     if k.startswith("selection_kcorr_c")})
        if _fid:
            print(f"    - Q_LSS build fiducials: {_fid}")
        # Carry the build-time conditioning values out so the CLI can enforce
        # them against the parameter space once it is known (the sampled-label
        # set does not exist yet at load time).  See
        # ``inference.q_provenance.check_lss_completion_provenance``.
        lss_completion_fiducials = {
            "path": lss_path,
            **{k: float(v) for k, v in _fid.items()},
        }
        print(
            "    [!] Q_LSS is FIXED at its build-time fiducials (cosmology, n0, "
            "delta, bias). Q is a radial completion field on the SAME zgrid "
            "(validated), interpreted as a dimensionless density-ratio. The "
            "parameters it is conditioned on may not be sampled — see the "
            "provenance check below."
        )
    return {
        "lss_completion_logq": lss_completion_logq,
        "lss_completion_logq_members": lss_completion_logq_members,
        "lss_completion_fiducials": lss_completion_fiducials,
        "lss_completion_indexing": lss_completion_indexing,
        "lss_completion_provenance": lss_completion_provenance,
    }

"""Preflight validation for spectral-siren strong-lensing inference inputs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from darksirens.lensing.observed_catalog import (
    observed_catalog_metadata_from_hdf5,
    validate_observed_catalog_file,
)
from darksirens.lensing.marginal_diagnostics import (
    candidate_time_mark_suspicion,
    resolve_sis_time_mark_impl,
    sis_time_mark_support,
    sis_time_mark_support_message,
)
from darksirens.lensing.lensed_injections import (
    DEFAULT_PAIR_ORIENTATION_MODE,
    pair_orientation_mismatch_message,
    read_pair_orientation_mode,
)
from darksirens.lensing.slmarks import DEFAULT_T0_SECONDS
from darksirens.lensing.partitions import (
    TIME_DERIVED_EDGE_MARK_KEYS,
    apply_edge_mark_prior_keys,
    connected_components_from_candidate_pairs,
    enumerate_compatible_partitions,
    exact_partition_components,
    parse_edge_mark_keys,
    parse_folded_mark_keys,
    validate_candidate_pairs,
)
from darksirens.likelihood.pair_kde import validate_pair_prior_wt
from darksirens.lensing.pair_tag_selection import make_pair_tag_selection_model
from darksirens.lensing import file_contract


def _get(opts: Any, name: str, default=None):
    return getattr(opts, name, default)


def _flag(opts: Any, name: str) -> bool:
    """A CLI boolean override, tolerating the string spellings argparse accepts.

    Library callers hand preflight plain SimpleNamespaces, so an override may
    arrive as "true" rather than True.
    """
    from darksirens.cli.common import str_to_bool

    value = _get(opts, name, False)
    try:
        return bool(str_to_bool(value))
    except ValueError:
        return False


def _read_json(path: str | Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _exists(path, errors, label):
    if not path:
        errors.append(f"missing {label}")
        return False
    if not Path(path).exists():
        errors.append(f"{label} does not exist: {path}")
        return False
    return True


def _infer_gw(path, errors, summary):
    if not _exists(path, errors, "gw_path"):
        return None, None
    try:
        with h5py.File(path, "r") as f:
            n_events = int(f.attrs.get("nobs", f.attrs.get("n_events", -1)))
            nsamp = int(f.attrs.get("nsamp", -1))
            if n_events < 0 and "m1det" in f and nsamp > 0:
                n_events = int(len(f["m1det"]) // nsamp)
            if nsamp < 0 and n_events > 0 and "m1det" in f:
                nsamp = int(len(f["m1det"]) // n_events)
            if n_events >= 0:
                summary["n_events"] = n_events
            if nsamp >= 0:
                summary["nsamp"] = nsamp
            return (n_events if n_events >= 0 else None), (
                nsamp if nsamp >= 0 else None
            )
    except Exception as exc:
        errors.append(f"gw_path not readable: {path}: {exc}")
        return None, None


def _resolve_observed_catalog(opts, gw_n_events, errors, warnings, summary):
    catalog_path = _get(opts, "observed_catalog_path")
    meta = None
    if catalog_path:
        try:
            meta = validate_observed_catalog_file(catalog_path)
        except Exception as exc:
            errors.append(f"observed_catalog_path invalid: {exc}")
            return None
        summary["observed_catalog"] = {k: v for k, v in meta.items() if k != "sha256"}
        if gw_n_events is not None and int(meta["n_events"]) != int(gw_n_events):
            errors.append(
                f"observed_catalog n_events={meta['n_events']} does not match gw n_events={gw_n_events}"
            )
        return meta
    gw_path = _get(opts, "gw_path")
    if gw_path and Path(gw_path).exists():
        try:
            h5_meta = observed_catalog_metadata_from_hdf5(gw_path)
        except Exception as exc:
            errors.append(f"gw_path observed-catalog attrs not readable: {exc}")
            return None
        if h5_meta is not None:
            summary["observed_catalog"] = h5_meta
            if gw_n_events is not None and int(h5_meta["n_events"]) != int(gw_n_events):
                errors.append(
                    f"observed PE n_events={h5_meta['n_events']} does not match gw n_events={gw_n_events}"
                )
            return h5_meta
    return None


def fixed_partition_coverage_error(used, n_events, *, path=None):
    """Return an error string if a fixed partition does not cover every event.

    A fixed partition must be a partition OF THE OBSERVED CATALOG: every event
    index must appear exactly once, as a singleton or as one endpoint of a pair.
    Shape, count self-consistency, index range and duplicates were all checked;
    COVERAGE was not, and the cluster likelihood only ever touches the listed
    indices.  An event in neither list is therefore silently excluded from the
    product while the selection correction uses the reduced counts -- a
    self-consistent inference on a SUBSET, reported over the PE file's event
    count (results.hdf5 n_events comes from the PE file, the reference partition
    counts from the partition).  A partition.json generated against an earlier or
    truncated mock listing 0..89 against a 100-event PE file passed every check
    and quietly analysed 90 events.  Returns ``None`` when the partition covers
    everything or when the event count is unknown.
    """
    if n_events is None:
        return None
    limit = int(n_events)
    missing = sorted(set(range(limit)) - {int(i) for i in used})
    if not missing:
        return None
    shown = ", ".join(str(i) for i in missing[:20])
    if len(missing) > 20:
        shown += f", ... (+{len(missing) - 20} more)"
    where = f" ({path})" if path else ""
    return (
        f"fixed partition does not cover every observed event{where}: "
        f"{len(missing)} of {limit} events appear in neither singleton_indices "
        f"nor pair_indices and would be SILENTLY DROPPED from the analysis "
        f"while the run still reports n_events={limit}. Missing: {shown}"
    )


def _check_partition(path, n_events, errors, summary, *, observed_n_events=None):
    if not _exists(path, errors, "partition_path"):
        return []
    try:
        data = _read_json(path)
    except Exception as exc:
        errors.append(f"partition_path not readable: {path}: {exc}")
        return []
    singletons = np.asarray(data.get("singleton_indices", []), dtype=int)
    pairs = np.asarray(data.get("pair_indices", []), dtype=int)
    if pairs.size == 0:
        pairs = pairs.reshape((0, 2))
    if pairs.ndim != 2 or pairs.shape[1] != 2:
        errors.append(
            f"partition pair_indices shape must be (n_pairs, 2), got {pairs.shape}"
        )
    if int(data.get("n_singletons", len(singletons))) != len(singletons):
        errors.append("partition n_singletons does not match singleton_indices")
    if int(data.get("n_pairs", pairs.shape[0] if pairs.ndim == 2 else -1)) != (
        pairs.shape[0] if pairs.ndim == 2 else -1
    ):
        errors.append("partition n_pairs does not match pair_indices")
    used = list(map(int, singletons)) + [int(x) for x in pairs.reshape(-1)]
    limit = observed_n_events if observed_n_events is not None else n_events
    if limit is not None and any(i < 0 or i >= limit for i in used):
        errors.append(
            f"partition index out of range for observed catalog n_events={limit}"
        )
    if len(set(used)) != len(used):
        errors.append("fixed partition uses at least one event more than once")
    coverage_error = fixed_partition_coverage_error(used, limit)
    if coverage_error is not None:
        errors.append(coverage_error)
    summary["n_pairs_partition"] = int(pairs.shape[0]) if pairs.ndim == 2 else None
    return (
        [tuple(map(int, p)) for p in pairs.reshape((-1, 2))] if pairs.ndim == 2 else []
    )


def _sis_support(opts, delta_t_values, sigma_values):
    """SIS-support verdict for a set of time marks, resolved like the runtime.

    The fatal case is ``all_annihilated`` (every pair's whole mark measure
    outside y in (0, 1)), not the bare ``y* >= 1``: the delta-collapse mark
    integrates over its own Gaussian width and the quadrature mark never masks,
    so the run this used to abort could have a perfectly finite likelihood.
    """
    T0 = _get(opts, "sl_T0_sec", None) or DEFAULT_T0_SECONDS
    return sis_time_mark_support(
        delta_t_values,
        T0,
        sigma_delta_t_seconds=sigma_values,
        mark_impl=resolve_sis_time_mark_impl(
            _get(opts, "pair_time_mark_impl", "auto"), sigma_values, T0
        ),
    )


def _check_pair_pe(
    path,
    n_events,
    partition_pairs,
    opts,
    errors,
    warnings,
    summary,
    *,
    unified_observed_mode=False,
    label="pair_pe_path",
):
    if not path:
        if unified_observed_mode and _get(opts, "pair_marks", "none") == "none":
            summary["pair_pe_metadata_optional"] = True
            return
        errors.append(f"missing {label}")
        return
    if not Path(path).exists():
        errors.append(f"{label} does not exist: {path}")
        return
    try:
        with h5py.File(path, "r") as f:
            if "npairs" not in f.attrs:
                errors.append(f"{label} missing npairs attribute")
                npairs = 0
            else:
                npairs = int(f.attrs["npairs"])
            summary["n_pairs_pair_pe"] = npairs
            pair_dt_values = []
            pair_sigma_values = []
            for k in range(npairs):
                pname = f"pair_{k}"
                if pname not in f:
                    errors.append(f"{label} missing group {pname}")
                    continue
                g = f[pname]
                if ("event_index_image0" in g.attrs) != (
                    "event_index_image1" in g.attrs
                ):
                    errors.append(
                        f"{pname} must define both event_index_image0 and event_index_image1"
                    )
                elif "event_index_image0" in g.attrs:
                    pair = (
                        int(g.attrs["event_index_image0"]),
                        int(g.attrs["event_index_image1"]),
                    )
                    if (
                        pair[0] == pair[1]
                        or pair[0] < 0
                        or pair[1] < 0
                        or (
                            n_events is not None
                            and (pair[0] >= n_events or pair[1] >= n_events)
                        )
                    ):
                        errors.append(
                            f"{pname} event-index metadata out of range: {pair}"
                        )
                    if (
                        partition_pairs
                        and k < len(partition_pairs)
                        and pair != partition_pairs[k]
                    ):
                        errors.append(
                            f"{pname} event-index metadata {pair} does not match partition pair {partition_pairs[k]}"
                        )
                has_dt = "delta_t_obs" in g.attrs or "delta_t_obs" in g
                # The runtime consumes per-pair marks POSITIONALLY against the
                # partition's pair rows, and the only order check is the
                # event-index one above -- which the optional attrs can switch
                # off entirely. Then a metadata file whose rows are ordered
                # differently from partition.json silently hands every pair
                # another pair's |dt| (wrong y* = |dt|/T0 and wrong 1/p_U(dt)
                # coincidence factor), with only a length check to catch it
                # (review F-014).
                if (
                    has_dt
                    and "event_index_image0" not in g.attrs
                    and _get(opts, "pair_marks", "none") == "time"
                    and _get(opts, "partition_mode", "fixed") == "fixed"
                    and partition_pairs
                ):
                    errors.append(
                        f"{pname} carries a time mark but no "
                        "event_index_image0/event_index_image1: with "
                        "--pair_marks time --partition_mode fixed the marks are "
                        "consumed in partition pair order and nothing would "
                        "verify the alignment"
                    )
                dt_recorded = False
                if has_dt:
                    try:
                        dt_value = float(
                            g.attrs["delta_t_obs"]
                            if "delta_t_obs" in g.attrs
                            else np.asarray(g["delta_t_obs"])[()]
                        )
                    except Exception as exc:
                        errors.append(f"{pname} delta_t_obs not readable: {exc}")
                    else:
                        pair_dt_values.append(dt_value)
                        # Kept aligned with pair_dt_values so the support
                        # diagnostic below can credit each mark's own width.
                        pair_sigma_values.append(None)
                        dt_recorded = True
                        # A non-finite delay is a broken mark, not a wide one:
                        # it propagates NaN/-inf through the marked pair
                        # likelihood, and the SIS support diagnostic below
                        # drops it rather than judging it.
                        if not np.isfinite(dt_value):
                            errors.append(
                                f"{pname} delta_t_obs must be finite, got {dt_value}"
                            )
                has_sig = (
                    "sigma_delta_t" in g.attrs
                    or "sigma_delta_t" in g
                    or _get(opts, "pair_time_sigma_sec") is not None
                )
                fixed_time_marks = (
                    _get(opts, "pair_marks", "none") == "time"
                    and _get(opts, "partition_mode", "fixed") == "fixed"
                )
                if fixed_time_marks:
                    if not has_dt:
                        errors.append(
                            f"pair_marks=time requires delta_t_obs metadata for {pname}"
                        )
                    if not has_sig:
                        errors.append(
                            f"pair_marks=time requires sigma_delta_t in {pname} or --pair_time_sigma_sec"
                        )
                if has_sig:
                    try:
                        sig = float(
                            g.attrs["sigma_delta_t"]
                            if "sigma_delta_t" in g.attrs
                            else (
                                np.asarray(g["sigma_delta_t"])[()]
                                if "sigma_delta_t" in g
                                else _get(opts, "pair_time_sigma_sec")
                            )
                        )
                        if not np.isfinite(sig) or sig <= 0:
                            errors.append(
                                f"{pname} sigma_delta_t must be finite and "
                                f"positive, got {sig}"
                            )
                        elif dt_recorded:
                            pair_sigma_values[-1] = sig
                    except Exception as exc:
                        errors.append(f"{pname} sigma_delta_t not readable: {exc}")
                if unified_observed_mode:
                    # Unified mode treats pair files as metadata only; image PE
                    # groups may be present for legacy compatibility but are
                    # never inference inputs.
                    if "image0" in g or "image1" in g:
                        warnings.append(
                            f"{label} contains legacy image PE groups in {pname}; "
                            "unified observed mode ignores them and reads PE samples from gw_path"
                        )
                    continue
                for img in ("image0", "image1"):
                    if img not in g:
                        errors.append(f"{pname} missing {img} group")
                        continue
                    gi = g[img]
                    for dset in ("m1det", "q", "dL_app", "chieff", "prior_wt"):
                        if dset not in gi:
                            errors.append(f"{pname}/{img} missing dataset {dset}")
                    if "prior_wt" in gi:
                        try:
                            validate_pair_prior_wt(
                                np.asarray(gi["prior_wt"]),
                                context=f"{pname}/{img}/prior_wt",
                            )
                        except Exception as exc:
                            errors.append(str(exc))
            if _get(opts, "pair_marks", "none") == "time" and pair_dt_values:
                # SIS support guard: a delay past y* = |dt|/T0 = 1 annihilates
                # the pair (-inf at every parameter value) only once the mark's
                # own width no longer reaches into the support -- and the
                # quadrature mark never annihilates at all.
                support = _sis_support(opts, pair_dt_values, pair_sigma_values)
                summary["sis_time_mark_support_pair_pe"] = support
                if support["all_annihilated"]:
                    errors.append(sis_time_mark_support_message(support))
                elif support["n_out_of_support"]:
                    warnings.append(sis_time_mark_support_message(support))
    except Exception as exc:
        errors.append(f"{label} not readable: {path}: {exc}")


def _check_candidates(path, n_events, opts, errors, warnings, summary, *, observed_n_events=None):
    if not _exists(path, errors, "candidate_pairs_path"):
        return
    try:
        data = _read_json(path)
        cand_n, pairs = validate_candidate_pairs(data)
        summary["n_candidate_pairs"] = len(pairs)
        expected = observed_n_events if observed_n_events is not None else n_events
        if expected is not None and cand_n != expected:
            errors.append(
                f"candidate_pairs n_events={cand_n} does not match gw n_events={n_events}"
            )
        available = sorted(
            set().union(*(p.marks.keys for p in pairs)) if pairs else set()
        )
        summary["available_edge_mark_keys"] = available
        prior_keys = parse_edge_mark_keys(_get(opts, "edge_mark_prior_keys"))
        for key in prior_keys:
            if not key.startswith("log_"):
                errors.append(
                    f"edge_mark_prior_keys only supports log_* marks, got {key!r}"
                )
            for pair in pairs:
                if pair.marks.get(key) is None:
                    errors.append(
                        f"candidate pair ({pair.i},{pair.j}) missing requested edge prior mark {key!r}"
                    )
        folded_keys = set(parse_folded_mark_keys(data))
        double_counted = sorted(k for k in prior_keys if k in folded_keys)
        if double_counted:
            errors.append(
                f"edge_mark_prior_keys {double_counted} are already folded into "
                f"log_prior_odds by the candidate-pair builder "
                f"(folded_mark_keys={sorted(folded_keys)}); applying them again "
                "would double-count the pairing prior"
            )
        like_keys = set(parse_edge_mark_keys(_get(opts, "edge_mark_likelihood_keys")))
        if _get(opts, "pair_marks", "none") == "time":
            like_keys.add("time")
        if like_keys & {"time", "delta_t_obs"}:
            # The double-count guard above only compares folded keys against the
            # requested PRIOR keys, so it cannot see the time term the builder
            # bakes into log_prior_odds -- yet the time edge-mark LIKELIHOOD
            # evaluates p(Delta t | y) on the very same delay. A pairing prior
            # tilted by |Delta t| plus that likelihood counts the arrival-time
            # separation twice, biasing the pairing posterior (and hence the
            # lensed fraction / A_tau) toward short separations.
            time_double_counted = sorted(
                k
                for k in (folded_keys | set(prior_keys))
                if k in TIME_DERIVED_EDGE_MARK_KEYS
            )
            if time_double_counted:
                errors.append(
                    f"time-derived edge marks {time_double_counted} enter the "
                    "pairing prior (folded_mark_keys="
                    f"{sorted(folded_keys)}, edge_mark_prior_keys="
                    f"{sorted(prior_keys)}) while the time edge-mark likelihood "
                    "scores the same delta_t_obs; that double-counts the "
                    "arrival-time separation. Rebuild candidate_pairs.json with "
                    "time marks exported (the builder then leaves the time score "
                    "out of log_prior_odds), or drop --pair_marks time"
                )
        unsupported_like = sorted(
            k for k in like_keys if k not in ("time", "delta_t_obs")
        )
        for key in unsupported_like:
            errors.append(
                f"edge_mark_likelihood_keys={key!r} is not implemented yet (only time is supported)"
            )
        if like_keys & {"time", "delta_t_obs"}:
            for pair in pairs:
                if pair.delta_t_obs is None or pair.sigma_delta_t is None:
                    errors.append(
                        "candidate pair "
                        f"({pair.i},{pair.j}) missing marks.delta_t_obs/sigma_delta_t "
                        "required by time edge-mark likelihood"
                    )
                elif not (
                    np.isfinite(float(pair.delta_t_obs))
                    and np.isfinite(float(pair.sigma_delta_t))
                ):
                    # Second net for pairs built in-process: the JSON path is
                    # already finiteness-validated by validate_edge_marks, but a
                    # library caller can construct CandidatePair directly, and
                    # the support diagnostic below silently discards non-finite
                    # delays instead of failing on them.
                    errors.append(
                        "candidate pair "
                        f"({pair.i},{pair.j}) has non-finite time marks: "
                        f"delta_t_obs={float(pair.delta_t_obs)}, "
                        f"sigma_delta_t={float(pair.sigma_delta_t)}"
                    )
            suspicion = candidate_time_mark_suspicion(pairs)
            summary.update(suspicion)
            if suspicion.get("candidate_time_marks_suspicious"):
                warnings.append(str(suspicion.get("candidate_time_marks_warning")))
            # SIS support guard: a delay past y* = |dt|/T0 = 1 annihilates the
            # pair (-inf at every parameter value) only once the mark's own
            # width no longer reaches into the support.
            support = _sis_support(
                opts,
                [p.delta_t_obs for p in pairs],
                [p.sigma_delta_t for p in pairs],
            )
            summary["sis_time_mark_support"] = support
            if support["all_annihilated"] and support["n_marked"] > 0:
                errors.append(sis_time_mark_support_message(support))
            elif support["n_out_of_support"]:
                warnings.append(sis_time_mark_support_message(support))
        effective_pairs = (
            apply_edge_mark_prior_keys(pairs, prior_keys) if not errors else pairs
        )
        components = connected_components_from_candidate_pairs(cand_n, effective_pairs)
        summary["n_components"] = len(components)
        summary["component_event_indices"] = [
            list(c["event_indices"]) for c in components
        ]
        summary["component_candidate_edge_indices"] = [
            list(c["candidate_edge_indices"]) for c in components
        ]
        component_mode = _get(opts, "partition_component_mode", "componentwise")
        summary["partition_component_mode"] = component_mode
        summary["component_sizes"] = [len(c["event_indices"]) for c in components]
        summary["component_edge_counts"] = [len(c["candidate_edge_indices"]) for c in components]
        if component_mode == "global":
            if len(components) > 1 and len(effective_pairs) > 8:
                summary["global_enumeration_warning"] = (
                    "global exact enumeration requested on a decomposable candidate graph; "
                    "componentwise mode is safer for larger simulations"
                )
            max_exact = int(_get(opts, "max_exact_partitions", 10000))
            global_cap = int(_get(opts, "max_total_partitions", max_exact) or max_exact)
            states = enumerate_compatible_partitions(
                cand_n,
                effective_pairs,
                max_partitions=global_cap,
            )
            summary["n_exact_partitions"] = len(states)
            summary["approximate_total_partitions"] = len(states)
            summary["largest_component_n_partitions"] = len(states)
            summary["factorized_exact_enabled"] = False
        else:
            max_exact = int(_get(opts, "max_exact_partitions", 10000))
            max_component_partitions = int(
                _get(opts, "max_component_partitions", max_exact) or max_exact
            )
            comp_summaries, component_states, approximate_total = exact_partition_components(
                cand_n,
                effective_pairs,
                max_component_events=_get(opts, "max_component_events"),
                max_component_edges=_get(opts, "max_component_edges"),
                max_component_partitions=max_component_partitions,
            )
            component_n_partitions = [c["n_partitions"] for c in comp_summaries]
            summary["component_n_partitions"] = component_n_partitions
            summary["n_exact_partitions"] = int(approximate_total)
            summary["approximate_total_partitions"] = int(approximate_total)
            summary["largest_component_n_partitions"] = int(max(component_n_partitions or [0]))
            summary["factorized_exact_enabled"] = True
            max_total = _get(opts, "max_total_partitions")
            if max_total is not None and int(approximate_total) > int(max_total):
                warnings.append(
                    "candidate_pairs approximate_total_partitions="
                    f"{int(approximate_total)} exceeds max_total_partitions={int(max_total)}; "
                    "componentwise factorized exact marginalization avoids global enumeration"
                )
    except Exception as exc:
        errors.append(f"candidate_pairs invalid: {exc}")


def _check_fixed_candidate_marks(
    path, partition_pairs, opts, errors, warnings, summary
):
    """Validate fixed-partition time marks that live in candidate_pairs.json.

    With a unified observed catalog, ``partition_mode=fixed`` and
    ``--pair_marks time``, the runtime builds the per-pair marks from the
    candidate graph when no pair file supplies them (the ``mark_by_edge``
    fallback in cli/inference_lensing, and a documented input mode).  Preflight
    demanded a pair PE/metadata file anyway, making that path unreachable, so
    waive the pair-file requirement when the candidate graph covers every
    partition edge -- and check those marks HERE, rather than leaving the run to
    exit on them later.  Returns True when the waiver applies.
    """
    if not _exists(path, errors, "candidate_pairs_path"):
        return False
    try:
        _, pairs = validate_candidate_pairs(_read_json(path))
    except Exception as exc:
        errors.append(f"candidate_pairs invalid: {exc}")
        return False
    mark_by_edge = {(p.i, p.j): p for p in pairs}
    marked = []
    missing = []
    for pair in partition_pairs:
        edge = tuple(sorted(int(x) for x in pair))
        cand = mark_by_edge.get(edge)
        if cand is None or cand.delta_t_obs is None or cand.sigma_delta_t is None:
            missing.append(edge)
            continue
        dt = float(cand.delta_t_obs)
        sig = float(cand.sigma_delta_t)
        if not np.isfinite(dt) or not np.isfinite(sig) or sig <= 0.0:
            errors.append(
                f"fixed partition pair {edge} has invalid candidate_pairs.json "
                f"time marks: delta_t_obs={dt}, sigma_delta_t={sig}"
            )
            continue
        marked.append(cand)
    if missing:
        errors.append(
            "pair_marks=time requires delta_t_obs/sigma_delta_t for every fixed "
            f"partition pair, but {missing} carry no candidate_pairs.json marks "
            "and no pair metadata file was provided"
        )
        return False
    if not marked:
        return False
    suspicion = candidate_time_mark_suspicion(marked)
    summary.update(suspicion)
    if suspicion.get("candidate_time_marks_suspicious"):
        warnings.append(str(suspicion.get("candidate_time_marks_warning")))
    support = _sis_support(
        opts,
        [p.delta_t_obs for p in marked],
        [p.sigma_delta_t for p in marked],
    )
    summary["sis_time_mark_support"] = support
    if support["all_annihilated"] and support["n_marked"] > 0:
        errors.append(sis_time_mark_support_message(support))
    elif support["n_out_of_support"]:
        warnings.append(sis_time_mark_support_message(support))
    summary["fixed_pair_marks_from_candidate_pairs"] = True
    return True


def _check_lensed(path, errors, summary, opts=None):
    if not _exists(path, errors, "lensed_injections_path"):
        return
    try:
        with h5py.File(path, "r") as f:
            n = int(f.attrs.get("Ndraw_sources", f.attrs.get("n_draw_sources", -1)))
            if n <= 0:
                errors.append(
                    "lensed_injections n_draw_sources/Ndraw_sources must be positive"
                )
            ptag_present = False
            for name, is_log in (
                ("log_p_tag_per_source", True),
                ("p_tag_per_source", False),
                ("log_p_tag", True),
                ("p_tag", False),
            ):
                if name in f:
                    ptag_present = True
                    arr = np.asarray(f[name], dtype=float)
                    if is_log and (
                        not np.all(np.isfinite(arr) | np.isneginf(arr))
                        or np.any(arr > 0)
                    ):
                        errors.append(
                            f"{name} must be finite/-inf and <= 0 in log-space"
                        )
                    if not is_log and (
                        not np.all(np.isfinite(arr)) or np.any((arr < 0) | (arr > 1))
                    ):
                        errors.append(f"{name} must be finite and in [0, 1]")
            summary["p_tag_present"] = ptag_present
            if not ptag_present:
                summary["p_tag_default"] = 1
            kind = _get(opts, "pair_tag_model", "constant") if opts is not None else "constant"
            summary["pair_tag_model"] = kind
            summary["pair_tag_perturb_logit"] = float(_get(opts, "pair_tag_perturb_logit", 0.0) or 0.0) if opts is not None else 0.0
            if kind == "file":
                if not _get(opts, "pair_tag_selection_path"):
                    errors.append("pair_tag_model=file requires --pair_tag_selection_path")
            elif kind != "constant":
                model = make_pair_tag_selection_model(kind, constant=float(_get(opts, "pair_tag_constant", 1.0)), perturb_logit=float(_get(opts, "pair_tag_perturb_logit", 0.0) or 0.0))
                for req in model.required_fields:
                    # load_lensed_injections resolves the delay as
                    # "delta_t_obs if present else true_delta_t", so a campaign
                    # that stores only true_delta_t scores fine and must not be
                    # refused here. Validate the dataset the LOADER would pick,
                    # in the loader's preference order -- checking the finite
                    # true_delta_t of a file whose delta_t_obs is all-NaN would
                    # pass a configuration the run then exits on.
                    candidates = (
                        ("delta_t_obs", "true_delta_t") if req == "delta_t_obs" else (req,)
                    )
                    dname = next((c for c in candidates if c in f), None)
                    if dname is None:
                        errors.append(
                            f"pair_tag_model={kind} requires lensed injection dataset "
                            + " or ".join(candidates)
                        )
                    else:
                        arr = np.asarray(f[dname], dtype=float)
                        if arr.size == 0 or np.all(~np.isfinite(arr)):
                            errors.append(f"pair_tag_model={kind} requires finite values in {dname}")
    except Exception as exc:
        errors.append(f"lensed_injections_path not readable: {path}: {exc}")


def _check_pair_orientation_mode(opts, errors, warnings, summary):
    """Compare the campaign's rendered orientation convention to the runtime one.

    The runtime ``pair_orientation_mode`` selects the lensed-singleton censoring
    factor while the campaign's rendered detection flags set the pair
    efficiencies; mixing the two conventions mis-normalises the
    J=2/lensed-singleton channel ratio that sets A_tau by up to ~2.6x.  It is
    an ERROR exactly when the lensed-singleton channel is enabled (and not
    explicitly waived), a warning otherwise -- mirroring the runtime gate in
    cli/inference_lensing._gate_pair_orientation_mismatch, so --preflight_only
    cannot pass a configuration the run itself would refuse.

    Deliberately NOT nested inside the cluster_mode=j2 block: the Mould-style
    per-event ablation is cluster_mode=off + --singleton_lensing sl_mixture,
    which is precisely the case where the mismatch bites.
    """
    path = _get(opts, "lensed_injections_path")
    if not path or not Path(path).exists():
        return
    try:
        campaign_mode = read_pair_orientation_mode(path)
    except Exception as exc:
        errors.append(
            f"lensed_injections_path pair_orientation_mode not readable: {path}: {exc}"
        )
        return
    run_mode = str(
        _get(opts, "pair_orientation_mode", DEFAULT_PAIR_ORIENTATION_MODE)
        or DEFAULT_PAIR_ORIENTATION_MODE
    )
    channel_enabled = _get(opts, "singleton_lensing", "off") == "sl_mixture"
    summary["pair_orientation_mode_campaign"] = campaign_mode
    summary["pair_orientation_mode_runtime"] = run_mode
    summary["pair_orientation_mode_match"] = bool(campaign_mode == run_mode)
    if campaign_mode == run_mode:
        return
    message = pair_orientation_mismatch_message(path, campaign_mode, run_mode)
    if channel_enabled and not _flag(opts, "allow_pair_orientation_mismatch"):
        errors.append(message)
    elif channel_enabled:
        warnings.append(f"{message} Waived by allow_pair_orientation_mismatch.")
    else:
        warnings.append(
            f"{message} (singleton_lensing=off, so the censoring factor is not "
            "evaluated in this run.)"
        )


def run_lensing_preflight(opts) -> dict:
    errors: list[str] = []
    warnings: list[str] = []
    summary = {
        "cluster_mode": _get(opts, "cluster_mode"),
        "partition_mode": _get(opts, "partition_mode", "fixed"),
        "partition_component_mode": _get(opts, "partition_component_mode", "componentwise"),
        "pair_marks": _get(opts, "pair_marks", "none"),
        "edge_mark_prior_keys": list(
            parse_edge_mark_keys(_get(opts, "edge_mark_prior_keys"))
        ),
        "edge_mark_likelihood_keys": list(
            parse_edge_mark_keys(_get(opts, "edge_mark_likelihood_keys"))
        ),
        "p_tag_present": False,
    }
    gw_report = file_contract.validate_observed_gw_pe(_get(opts, "gw_path")) if _get(opts, "gw_path") else {"ok": False, "errors": ["missing gw_path"], "summary": {}}
    summary["file_contract"] = {"observed_gw_pe": gw_report}
    if gw_report.get("ok"):
        summary.update({k: v for k, v in gw_report["summary"].items() if k in ("n_events", "nsamp")})
        n_events = gw_report["summary"].get("n_events")
    else:
        n_events, _ = _infer_gw(_get(opts, "gw_path"), errors, summary)
        errors.extend([f"observed_gw_pe: {e}" for e in gw_report.get("errors", []) if "format_version" not in e])
        if any("format_version" in e for e in gw_report.get("errors", [])):
            warnings.extend([f"observed_gw_pe contract warning: {e}" for e in gw_report.get("errors", [])])
    def _adopt_report(name, report):
        """Store a file-contract sub-report AND surface its errors.

        Reports used to be stashed in summary["file_contract"] without their
        errors reaching the top-level list, so a preflight could print OK
        while a stored report said otherwise (review P2-06).  Format-
        recognition failures stay WARNINGS (same split the observed-PE
        report gets above): the runtime loaders accept legacy files the
        contract does not know, and the contract must not fail inputs the
        run would load fine.
        """
        summary["file_contract"][name] = report
        if not report.get("ok", True):
            for e in report.get("errors", []):
                if "format_version" in e:
                    warnings.append(f"{name} contract warning: {e}")
                else:
                    errors.append(f"{name}: {e}")
        return report

    if _get(opts, "observed_catalog_path"):
        _adopt_report(
            "observed_catalog",
            file_contract.validate_observed_catalog(
                _get(opts, "observed_catalog_path")
            ),
        )
    observed_meta = _resolve_observed_catalog(opts, n_events, errors, warnings, summary)
    observed_n_events = (
        int(observed_meta["n_events"]) if observed_meta is not None else None
    )
    _exists(_get(opts, "gwselection_path"), errors, "gwselection_path")
    if _get(opts, "gwselection_path"):
        _adopt_report(
            "gwselection",
            file_contract.validate_selection_inputs(_get(opts, "gwselection_path")),
        )
    if _get(opts, "selection_path"):
        _adopt_report(
            "selection_inputs",
            file_contract.validate_selection_inputs(_get(opts, "selection_path")),
        )
    partition_pairs = []
    unified_observed_mode = False
    if _get(opts, "cluster_mode") == "j2":
        if _get(opts, "partition_mode", "fixed") == "fixed":
            partition_pairs = _check_partition(
                _get(opts, "partition_path"),
                n_events,
                errors,
                summary,
                observed_n_events=observed_n_events,
            )
            if n_events is not None and partition_pairs:
                used = [i for pair in partition_pairs for i in pair]
                singletons = []
                try:
                    singletons = list(
                        map(
                            int,
                            _read_json(_get(opts, "partition_path")).get(
                                "singleton_indices", []
                            ),
                        )
                    )
                except Exception:
                    pass
                unified_observed_mode = max(
                    singletons + used, default=-1
                ) + 1 == n_events and (
                    _get(opts, "pair_metadata_path") or not _get(opts, "pair_pe_path")
                )
        elif _get(opts, "partition_mode") == "marginalize_exact":
            cand_report = file_contract.validate_candidate_pairs(_get(opts, "candidate_pairs_path"))
            summary["file_contract"]["candidate_pairs"] = cand_report
            summary["n_candidate_pairs"] = cand_report.get("summary", {}).get("n_candidate_pairs")
            warnings.extend([f"candidate_pairs: {w}" for w in cand_report.get("warnings", [])])
            _check_candidates(
                _get(opts, "candidate_pairs_path"),
                n_events,
                opts,
                errors,
                warnings,
                summary,
                observed_n_events=observed_n_events,
            )
            try:
                cand = _read_json(_get(opts, "candidate_pairs_path"))
                unified_observed_mode = n_events is not None and int(
                    cand.get("n_events", -1)
                ) == int(n_events)
            except Exception:
                unified_observed_mode = False
        if observed_meta is not None:
            unified_observed_mode = True
        elif unified_observed_mode:
            warnings.append(
                "unified observed mode was inferred by deprecated event-count heuristic"
            )
            summary["unified_observed_mode_inferred_heuristic"] = True
        summary["unified_observed_mode"] = bool(unified_observed_mode)
        if _get(opts, "lensed_injections_path"):
            summary["file_contract"]["lensed_injections"] = file_contract.validate_selection_inputs(_get(opts, "lensed_injections_path"))
        _check_lensed(_get(opts, "lensed_injections_path"), errors, summary, opts)
        pair_meta_path = _get(opts, "pair_metadata_path")
        pair_pe_path = _get(opts, "pair_pe_path")
        if (
            pair_meta_path
            and pair_pe_path
            and Path(pair_meta_path) != Path(pair_pe_path)
        ):
            errors.append(
                "--pair_metadata_path and --pair_pe_path both provided but point to different paths"
            )
        pair_path = pair_meta_path or pair_pe_path
        pair_label = "pair_metadata_path" if pair_meta_path else "pair_pe_path"
        fixed_marks_from_candidates = False
        if (
            unified_observed_mode
            and not pair_path
            and _get(opts, "partition_mode", "fixed") == "fixed"
            and _get(opts, "pair_marks", "none") == "time"
            and _get(opts, "candidate_pairs_path")
            and partition_pairs
        ):
            fixed_marks_from_candidates = _check_fixed_candidate_marks(
                _get(opts, "candidate_pairs_path"),
                partition_pairs,
                opts,
                errors,
                warnings,
                summary,
            )
        if not (
            unified_observed_mode
            and _get(opts, "candidate_pairs_path")
            and (
                _get(opts, "partition_mode", "fixed") == "marginalize_exact"
                or fixed_marks_from_candidates
            )
        ):
            _check_pair_pe(
                pair_path,
                n_events,
                partition_pairs,
                opts,
                errors,
                warnings,
                summary,
                unified_observed_mode=unified_observed_mode,
                label=pair_label,
            )
        elif pair_path:
            _check_pair_pe(
                pair_path,
                n_events,
                partition_pairs,
                opts,
                errors,
                warnings,
                summary,
                unified_observed_mode=unified_observed_mode,
                label=pair_label,
            )
        else:
            summary["pair_metadata_from_candidate_pairs"] = True
    _check_pair_orientation_mode(opts, errors, warnings, summary)
    if (
        _get(opts, "wl_selection") == "wl_lognormal"
        and _get(opts, "wl_backend") != "lognormal"
    ):
        warnings.append(
            "wl_selection=wl_lognormal is only meaningful with wl_backend=lognormal"
        )
    if _get(opts, "wl_backend") == "tabulated":
        table_path = _get(opts, "lensing_wl_table_path")
        if not table_path:
            errors.append(
                "wl_backend=tabulated requires lensing_wl_table_path"
            )
        else:
            _exists(table_path, errors, "lensing_wl_table_path")
    if _get(opts, "fix_lens_rate", True):
        if (
            not np.isfinite(float(_get(opts, "sl_tau_A", 0)))
            or float(_get(opts, "sl_tau_A", 0)) <= 0
        ):
            errors.append("sl_tau_A must be positive when lens rate is fixed")
        if not np.isfinite(float(_get(opts, "sl_tau_n", np.nan))):
            errors.append("sl_tau_n must be finite")
    elif _get(opts, "lens_prior_overrides"):
        try:
            over = (
                json.loads(_get(opts, "lens_prior_overrides"))
                if isinstance(_get(opts, "lens_prior_overrides"), str)
                else _get(opts, "lens_prior_overrides")
            )
            for key, bounds in over.items():
                lo, hi = map(float, bounds)
                if not np.isfinite(lo) or not np.isfinite(hi) or lo >= hi:
                    errors.append(f"invalid lens_prior_overrides bounds for {key}")
        except Exception as exc:
            errors.append(f"lens_prior_overrides must be valid JSON bounds: {exc}")
    return {
        "ok": not errors,
        "errors": errors,
        "warnings": warnings,
        "summary": summary,
    }

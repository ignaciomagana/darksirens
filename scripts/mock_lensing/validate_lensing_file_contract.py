#!/usr/bin/env python
"""Validate the unified lensing file contract."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from darksirens.lensing import file_contract


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gw_path", required=True)
    p.add_argument("--observed_catalog_path", required=True)
    p.add_argument("--candidate_pairs_path", required=True)
    p.add_argument("--selection_path", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--out")
    args = p.parse_args(argv)
    report = {
        "observed_gw_pe": file_contract.validate_observed_gw_pe(args.gw_path),
        "observed_catalog": file_contract.validate_observed_catalog(args.observed_catalog_path),
        "candidate_pairs": file_contract.validate_candidate_pairs(args.candidate_pairs_path),
        "selection_inputs": file_contract.validate_selection_inputs(args.selection_path),
        "run_config": file_contract.validate_run_config(args.config),
    }
    report["ok"] = all(v.get("ok", False) for v in report.values() if isinstance(v, dict))
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.out:
        Path(args.out).write_text(text)
    else:
        print(text, end="")
    return 0 if report["ok"] else 1

if __name__ == "__main__":
    raise SystemExit(main())

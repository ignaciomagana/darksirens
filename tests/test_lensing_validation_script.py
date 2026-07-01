import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.integration
@pytest.mark.slow
def test_lensing_validation_tiny_diagnostics(tmp_path):
    repo = Path(__file__).resolve().parents[1]
    cmd = [
        sys.executable,
        "scripts/mock_lensing/run_lensing_validation.py",
        "--profile",
        "tiny",
        "--workdir",
        str(tmp_path / "ds_lensing_validation"),
    ]
    subprocess.run(cmd, cwd=repo, check=True, timeout=300)

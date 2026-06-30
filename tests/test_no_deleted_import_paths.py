from pathlib import Path

FORBIDDEN = [
    "darksirens.em",
    "darksirens.tool",
    "darksirens.utils.containers",
    "darksirens.inference.likelihood",
    "darksirens.inference.likelihood_core",
    "darksirens/inference/catalog_views.py",
    "darksirens/utils/containers.py",
]

ROOTS = [Path("darksirens"), Path("scripts")]


def test_deleted_import_paths_not_used_in_production_code():
    hits = []
    for root in ROOTS:
        if not root.exists():
            continue
        for path in root.rglob("*.py"):
            text = path.read_text(encoding="utf-8")
            for needle in FORBIDDEN:
                if needle in text:
                    hits.append(f"{path}: contains {needle!r}")
    assert not hits, "Deleted legacy paths found:\n" + "\n".join(hits)

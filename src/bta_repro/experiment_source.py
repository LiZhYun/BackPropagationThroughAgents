"""Read a source snapshot before an experiment creates any output files."""

import hashlib
import json
from pathlib import Path


def source_snapshot(root, source_manifest=None, *, required_modules=("__init__.py", "simulator_train.py"),
                    module_names=None):
    """Return verified bytes, hashes and provenance for a local experiment.

    A working-tree snapshot may select a minimal module set. A frozen snapshot
    is always copied in full and must match its manifest exactly, including the
    file set. This function reads files; the caller owns destination creation.
    """
    root = Path(root).resolve()
    if source_manifest is None:
        source = root / "src/bta_repro"
        provenance = {"kind": "working_tree", "path": "src/bta_repro"}
        expected = None
    else:
        if module_names is not None:
            raise ValueError("a frozen source snapshot must be copied in full")
        source_manifest = Path(source_manifest).resolve()
        if not source_manifest.is_relative_to(root):
            raise ValueError("source manifest must belong to this repository")
        manifest_bytes = source_manifest.read_bytes()
        expected = json.loads(manifest_bytes)["source_sha256"]
        source = source_manifest.parent / "source/bta_repro"
        provenance = {"kind": "frozen_manifest", "manifest": str(source_manifest.relative_to(root)),
                      "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest()}
    if not source.resolve().is_relative_to(root):
        raise ValueError("source directory must belong to this repository")
    paths = sorted(source.glob("*.py"))
    if module_names is not None:
        if any(Path(name).name != name or not name.endswith(".py") for name in module_names):
            raise ValueError("source module names must be Python basenames")
        paths = [source / name for name in sorted(set(module_names))]
    files = {}
    for path in paths:
        if not path.resolve().is_relative_to(source.resolve()):
            raise ValueError("source file points outside the snapshot")
        files[path.name] = path.read_bytes()
    if not set(required_modules).issubset(files):
        raise ValueError("source snapshot is missing a required package module")
    hashes = {name: hashlib.sha256(data).hexdigest() for name, data in files.items()}
    if expected is not None and hashes != expected:
        raise ValueError("frozen source hashes differ from the selected manifest")
    return files, hashes, provenance

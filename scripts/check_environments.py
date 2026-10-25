#!/usr/bin/env python3
"""Print JSON runtime provenance without constructing a simulator."""
import argparse
from contextlib import redirect_stdout
import importlib
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import sys

PROJECT = Path(__file__).resolve().parents[1]


def package(name):
    try:
        dist = metadata.distribution(name)
    except metadata.PackageNotFoundError:
        return {"version": None}
    direct = json.loads(dist.read_text("direct_url.json") or "{}")
    return {"version": dist.version, "installer": (dist.read_text("INSTALLER") or "").strip(),
            "source_commit": direct.get("vcs_info", {}).get("commit_id"),
            "editable": direct.get("dir_info", {}).get("editable", False)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("core", "smac", "mujoco"), default="core")
    parser.add_argument("--strict", action="store_true", help="fail on pin mismatches")
    parser.add_argument("--imports", action="store_true", help="import modules without starting engines")
    parser.add_argument("--sc2-path", help="inspect existing SC2 files; otherwise use SC2PATH for smac")
    args = parser.parse_args()
    sys.dont_write_bytecode = True
    os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"
    expected = {"numpy": None, "torch": "2.6.0+cpu", "bta-repro": "0.1.0"}
    python_pin, commit = None, None
    if args.profile == "smac":
        lock = json.loads((PROJECT / "configs/environment_locks/smac-cpu.json").read_text())
        expected, python_pin = lock["packages"], lock["python"]
        commit = ("smacv2", lock["smacv2_commit"])
    elif args.profile == "mujoco":
        expected.update({"numpy": "1.23.5", "gym": "0.10.8", "mujoco-py": "2.1.2.14", "Cython": "0.29.36", "MAMujoco": "1.1.0"})
        python_pin = "3.10"
        commit = ("MAMujoco", "b212ddd74b258e7cea006ff1d642b5ffada4b99d")
    packages = {name: package(name) for name in expected}
    mismatches, errors = [], []
    for name, pin in expected.items():
        actual = packages[name]["version"]
        if actual is None or (pin is not None and actual != pin):
            mismatches.append(f"{name}: expected {pin or 'installed'}, found {actual}")
    if python_pin and not (platform.python_version() == python_pin or platform.python_version().startswith(python_pin + ".")):
        mismatches.append(f"Python: expected {python_pin}, found {platform.python_version()}")
    if commit and packages[commit[0]].get("source_commit") != commit[1]:
        mismatches.append(f"{commit[0]} source commit differs from {commit[1]}")
    report = {"schema_version": 1, "profile": args.profile, "purpose": "inspection only; no simulator episodes",
              "python": {"version": platform.python_version(), "executable": sys.executable,
                         "prefix": sys.prefix, "base_prefix": sys.base_prefix},
              "host": {"system": platform.system(), "machine": platform.machine(), "libc": platform.libc_ver()},
              "packages": packages, "pin_mismatches": mismatches, "errors": errors}
    if args.imports:
        modules = ["numpy", "torch"] + (["smacv2.env.starcraft2.wrapper"] if args.profile == "smac" else ["gym"] if args.profile == "mujoco" else [])
        report["imports"] = {}
        for name in modules:
            try:
                with redirect_stdout(sys.stderr):
                    imported = importlib.import_module(name)
                report["imports"][name] = "passed"
                if name == "torch":
                    report["torch_cuda_build"] = imported.version.cuda
                    if imported.version.cuda is not None:
                        mismatches.append("CPU profile expected; Torch contains a CUDA build")
            except Exception as exc:
                errors.append(f"{name}: {type(exc).__name__}: {exc}")
                report["imports"][name] = "failed"
        if args.profile == "mujoco":
            report["imports"]["mujoco_py"] = "metadata only: import can compile; use separate simulator smoke check"
    sc2 = args.sc2_path or (os.environ.get("SC2PATH") if args.profile == "smac" else None)
    report["engine"] = {"status": "not_requested"}
    if sc2:
        root = Path(sc2).expanduser().resolve()
        binary = root / "Versions/Base75689/SC2_x64"
        engine = {"path": str(root), "expected_version": "4.10", "build": "Base75689", "binary_exists": binary.is_file(), "maps": {}}
        if not binary.is_file():
            errors.append(f"missing SC2 binary {binary}")
        for race in ("protoss", "terran", "zerg"):
            name = f"10gen_{race}.SC2Map"
            found = [str(p) for p in (root / "Maps" / name, root / "Maps/SMAC_Maps" / name) if p.is_file()]
            engine["maps"][name] = found
            if not found:
                errors.append(f"missing SC2 map {name}")
        report["engine"] = engine
    elif args.profile == "mujoco" and os.environ.get("MUJOCO_PY_MUJOCO_PATH"):
        root = Path(os.environ["MUJOCO_PY_MUJOCO_PATH"])
        report["engine"] = {"path": str(root), "library_exists": (root / "bin/libmujoco210.so").is_file()}
        source = root / "SOURCE.json"
        if source.is_file():
            report["engine"]["archive_provenance"] = json.loads(source.read_text())
        if not report["engine"]["library_exists"]:
            errors.append(f"missing MuJoCo library in {root}")
    report["ok"] = not errors and not mismatches
    print(json.dumps(report, indent=2, sort_keys=True))
    return int(bool(errors or (args.strict and mismatches)))


if __name__ == "__main__":
    raise SystemExit(main())

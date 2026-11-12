#!/usr/bin/env python3
"""
scripts/setup_check.py
Environment verification for the AI Garment Multi-Label Size Inspection System.

Checks:
  - Python >= 3.10
  - CUDA availability + GPU name
  - All required packages importable
  - config.yaml present and parseable
  - Required data directories exist

Prints a formatted ASCII table and exits with code 0 (all pass) or 1 (any fail).

Usage::

    python scripts/setup_check.py
"""
from __future__ import annotations

import importlib
import os
import platform
import sys
from pathlib import Path
from typing import List, Tuple

# ---------------------------------------------------------------------------
# Repo root (two levels up from this script)
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------------------
# Packages to verify (import name, pip name shown in notes)
# ---------------------------------------------------------------------------
REQUIRED_PACKAGES: List[Tuple[str, str]] = [
    ("ultralytics", "ultralytics"),
    ("torch", "torch"),
    ("torchvision", "torchvision"),
    ("paddle", "paddlepaddle-gpu"),
    ("paddleocr", "paddleocr"),
    ("cv2", "opencv-python"),
    ("faiss", "faiss-cpu"),
    ("fastapi", "fastapi"),
    ("uvicorn", "uvicorn"),
    ("PyQt6", "PyQt6"),
    ("sqlalchemy", "sqlalchemy"),
    ("numpy", "numpy"),
    ("PIL", "pillow"),
    ("pydantic", "pydantic"),
    ("yaml", "pyyaml"),
    ("serial", "pyserial"),
    ("scipy", "scipy"),
    ("sklearn", "scikit-learn"),
    ("aiofiles", "aiofiles"),
    ("multipart", "python-multipart"),
    ("httpx", "httpx"),
    ("colorama", "colorama"),
]

REQUIRED_DIRS: List[str] = [
    "data/reference_images",
    "data/snapshots",
    "models/label_detector/weights",
    "models/feature_extractor",
    "models/ocr/det",
    "models/ocr/rec",
]

# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

class CheckResult:
    def __init__(self, name: str, ok: bool, note: str = ""):
        self.name = name
        self.ok = ok
        self.note = note


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------

def check_python() -> CheckResult:
    ver = sys.version_info
    ok = ver >= (3, 10)
    note = f"{ver.major}.{ver.minor}.{ver.micro}"
    return CheckResult("Python >= 3.10", ok, note if ok else f"{note} — NEED 3.10+")


def check_cuda() -> CheckResult:
    try:
        import torch  # type: ignore

        available = torch.cuda.is_available()
        if available:
            name = torch.cuda.get_device_name(0)
            mem_gb = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
            note = f"{name} ({mem_gb:.1f} GB)"
        else:
            note = "CUDA not available — will use CPU"
        return CheckResult("CUDA / GPU", available, note)
    except ImportError:
        return CheckResult("CUDA / GPU", False, "torch not installed")


def check_packages() -> List[CheckResult]:
    results = []
    for import_name, pip_name in REQUIRED_PACKAGES:
        try:
            mod = importlib.import_module(import_name)
            ver = getattr(mod, "__version__", "?")
            results.append(CheckResult(f"pkg: {pip_name}", True, f"v{ver}"))
        except ImportError:
            results.append(
                CheckResult(f"pkg: {pip_name}", False, f"pip install {pip_name}")
            )
    return results


def check_config() -> CheckResult:
    cfg_path = REPO_ROOT / "config.yaml"
    if not cfg_path.exists():
        return CheckResult("config.yaml", False, f"Not found at {cfg_path}")
    try:
        import yaml  # type: ignore

        with open(cfg_path, encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        keys = ", ".join(data.keys()) if isinstance(data, dict) else "?"
        return CheckResult("config.yaml", True, f"sections: [{keys}]")
    except Exception as exc:
        return CheckResult("config.yaml", False, str(exc))


def check_dirs() -> List[CheckResult]:
    results = []
    for rel in REQUIRED_DIRS:
        p = REPO_ROOT / rel
        exists = p.exists()
        results.append(
            CheckResult(
                f"dir: {rel}",
                exists,
                "OK" if exists else "mkdir -Force to create",
            )
        )
    return results


# ---------------------------------------------------------------------------
# Table printer
# ---------------------------------------------------------------------------

ANSI_GREEN = "\033[92m"
ANSI_RED = "\033[91m"
ANSI_RESET = "\033[0m"
ANSI_BOLD = "\033[1m"

try:
    import colorama  # type: ignore

    colorama.init()
except ImportError:
    pass


def _status(ok: bool) -> str:
    if ok:
        return f"{ANSI_GREEN}  PASS  {ANSI_RESET}"
    return f"{ANSI_RED}  FAIL  {ANSI_RESET}"


def print_table(results: List[CheckResult]) -> None:
    col_check = max(len(r.name) for r in results) + 2
    col_note = max(len(r.note) for r in results) + 2
    col_status = 8

    sep = f"+{'-' * col_check}+{'-' * col_status}+{'-' * col_note}+"
    header = (
        f"| {'Check':<{col_check - 2}} | {'Status':^{col_status - 2}} | "
        f"{'Note':<{col_note - 2}} |"
    )
    print(sep)
    print(f"{ANSI_BOLD}{header}{ANSI_RESET}")
    print(sep)
    for r in results:
        name_col = f" {r.name:<{col_check - 2}} "
        note_col = f" {r.note:<{col_note - 2}} "
        print(f"|{name_col}|{_status(r.ok)}|{note_col}|")
    print(sep)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    print(f"\n{ANSI_BOLD}AI Garment Inspection — Environment Check{ANSI_RESET}")
    print(f"Platform : {platform.system()} {platform.release()} ({platform.machine()})")
    print(f"Repo root: {REPO_ROOT}\n")

    all_results: List[CheckResult] = []
    all_results.append(check_python())
    all_results.append(check_cuda())
    all_results.extend(check_packages())
    all_results.append(check_config())
    all_results.extend(check_dirs())

    print_table(all_results)

    failures = [r for r in all_results if not r.ok]
    total = len(all_results)
    passed = total - len(failures)

    print(f"\nResult: {passed}/{total} checks passed", end="")
    if failures:
        print(f" — {ANSI_RED}{len(failures)} FAILED{ANSI_RESET}")
        print("\nFailed checks:")
        for r in failures:
            print(f"  ✗ {r.name}: {r.note}")
        return 1
    else:
        print(f" — {ANSI_GREEN}ALL OK{ANSI_RESET}")
        return 0


if __name__ == "__main__":
    sys.exit(main())

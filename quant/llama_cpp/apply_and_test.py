#!/usr/bin/env python3
"""Apply and verify the revision-locked TQ1_V llama.cpp integration."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Sequence


ROOT = Path(__file__).resolve().parent
REPOSITORY = ROOT.parents[1]
MANIFEST_PATH = ROOT / "integration.json"


def _run(command: Sequence[str | Path], *, cwd: Path,
         env: dict[str, str] | None = None) -> None:
    rendered = [str(item) for item in command]
    print("+", " ".join(rendered), flush=True)
    subprocess.run(rendered, cwd=cwd, env=env, check=True)


def _capture(command: Sequence[str | Path], *, cwd: Path) -> str:
    return subprocess.check_output(
        [str(item) for item in command], cwd=cwd, text=True).strip()


def _manifest() -> dict[str, object]:
    data = json.loads(MANIFEST_PATH.read_text())
    required = {
        "integration", "source", "patch", "patch_sha256",
        "spec_revision", "format_version", "ggml_type_registry_revision",
    }
    if set(data) < required:
        raise RuntimeError("incomplete llama.cpp integration manifest")
    return data


def _verify_inputs(target: Path, manifest: dict[str, object]) -> Path:
    if not target.is_dir():
        raise FileNotFoundError(target)
    if _capture(["git", "rev-parse", "--is-inside-work-tree"], cwd=target) != "true":
        raise RuntimeError(f"{target} is not a git worktree")
    expected_revision = str(manifest["source"]["revision"])  # type: ignore[index]
    observed_revision = _capture(["git", "rev-parse", "HEAD"], cwd=target)
    if observed_revision != expected_revision:
        raise RuntimeError(
            f"llama.cpp revision mismatch: {observed_revision} != {expected_revision}")
    dirty = _capture(["git", "status", "--porcelain"], cwd=target)
    if dirty:
        raise RuntimeError(
            "target llama.cpp worktree is dirty; use a clean clone/worktree so "
            "the integration remains reproducible")
    patch = ROOT / str(manifest["patch"])
    observed_patch_hash = hashlib.sha256(patch.read_bytes()).hexdigest()
    if observed_patch_hash != manifest["patch_sha256"]:
        raise RuntimeError(
            f"integration patch hash mismatch: {observed_patch_hash}")
    _run(["git", "apply", "--check", patch], cwd=target)
    return patch


def _build_and_test(target: Path, build: Path, jobs: int,
                    cxx: str, test_source: Path, *,
                    skip_model_test: bool) -> None:
    _run([
        "cmake", "-S", target, "-B", build,
        "-DGGML_METAL=OFF",
        "-DGGML_ACCELERATE=OFF",
        "-DGGML_BLAS=OFF",
        "-DGGML_OPENMP=OFF",
        "-DLLAMA_BUILD_TESTS=ON",
        "-DLLAMA_BUILD_EXAMPLES=OFF",
        "-DLLAMA_BUILD_TOOLS=OFF",
        "-DCMAKE_BUILD_TYPE=Release",
    ], cwd=target)
    _run([
        "cmake", "--build", build, "--parallel", str(jobs),
        "--target", "llama", "test-backend-ops",
    ], cwd=target)
    binary = build / "bin" / "test-tq1-v"
    library_dir = build / "bin"
    _run([
        cxx, "-std=c++17", "-O2", test_source,
        f"-I{target / 'ggml' / 'include'}",
        f"-L{library_dir}",
        f"-Wl,-rpath,{library_dir}",
        "-lggml", "-lggml-cpu", "-lggml-base", "-pthread",
        "-o", binary,
    ], cwd=target)
    env = dict(os.environ)
    for variable in ("DYLD_LIBRARY_PATH", "LD_LIBRARY_PATH"):
        previous = env.get(variable)
        env[variable] = str(library_dir) + (os.pathsep + previous if previous else "")
    _run([binary], cwd=target, env=env)
    if skip_model_test:
        return

    fixture_dir = build / "tq1-model-fixture"
    _run([
        sys.executable,
        ROOT / "tests" / "make_tiny_fixture.py",
        "--output", fixture_dir,
        "--converter", target / "convert_hf_to_gguf.py",
        "--overwrite",
    ], cwd=REPOSITORY)
    model_binary = build / "bin" / "test-tq1-model"
    _run([
        cxx, "-std=c++17", "-O2",
        ROOT / "tests" / "tq1_model_test.cpp",
        f"-I{target / 'include'}",
        f"-I{target / 'ggml' / 'include'}",
        f"-L{library_dir}",
        f"-Wl,-rpath,{library_dir}",
        "-lllama", "-lggml", "-lggml-cpu", "-lggml-base", "-pthread",
        "-o", model_binary,
    ], cwd=target)
    _run([
        model_binary, fixture_dir / "tiny-tq1-v11-r.gguf",
    ], cwd=target, env=env)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply the pinned BitNet TQ1_V patch to a clean llama.cpp clone")
    parser.add_argument(
        "--target", type=Path, required=True,
        help="explicit clean llama.cpp clone/worktree (the script has no implicit target)")
    parser.add_argument(
        "--build-dir", type=Path,
        help="CMake build directory (default: <target>/build-bitnet-tq1)")
    parser.add_argument("--jobs", type=int, default=max(1, os.cpu_count() or 1))
    parser.add_argument("--cxx", default=os.environ.get("CXX", "c++"))
    parser.add_argument(
        "--test-source", type=Path, default=ROOT / "tests" / "tq1_kernel_test.cpp")
    parser.add_argument(
        "--check-only", action="store_true",
        help="verify revision/hash/applicability without changing the target")
    parser.add_argument(
        "--skip-model-test", action="store_true",
        help="skip deterministic PTQ/GGUF/model-load test (low-level test still runs)")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.jobs < 1:
        raise ValueError("--jobs must be positive")
    target = args.target.expanduser().resolve()
    manifest = _manifest()
    patch = _verify_inputs(target, manifest)
    if args.check_only:
        print("Pinned llama.cpp revision and TQ1_V patch: CHECK PASS")
        return 0
    _run(["git", "apply", patch], cwd=target)
    build = (args.build_dir or target / "build-bitnet-tq1").expanduser().resolve()
    test_source = args.test_source.expanduser().resolve()
    if not test_source.is_file():
        raise FileNotFoundError(test_source)
    _build_and_test(
        target, build, args.jobs, args.cxx, test_source,
        skip_model_test=args.skip_model_test)
    print("Pinned llama.cpp TQ1_V integration: PASS")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, subprocess.CalledProcessError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1)

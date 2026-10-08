"""Run-folder helpers: make every experiment output folder self-contained.

`snapshot_runner` copies the exact runner script that produced a run (plus any
extra config files) into that run's output folder and writes a small manifest.
Together with the per-run convention
    artifacts/historical/<dataset>/<run_id>/
each run folder then contains code + config + metrics + model + predictions,
so an experiment can be inspected or rerun later without hunting for scripts.
"""
from __future__ import annotations

import datetime as _dt
import json
import shutil
import sys
from pathlib import Path


def snapshot_runner(output_dir, runner_path=None, extra_files=()):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    src = Path(runner_path or sys.argv[0]).resolve()
    saved = []
    if src.exists():
        dest = output_dir / f"RUNNER_{src.name}"
        shutil.copy2(src, dest)
        saved.append(dest.name)
    for extra in extra_files:
        extra = Path(extra)
        if extra.exists():
            shutil.copy2(extra, output_dir / extra.name)
            saved.append(extra.name)
    manifest = {
        "runner": src.name,
        "captured_at_utc": _dt.datetime.utcnow().isoformat(timespec="seconds"),
        "saved_files": saved,
    }
    (output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return output_dir

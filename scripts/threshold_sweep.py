"""
Threshold sweep for obj_threshold on moma_selfmotr_v5_nomotr.

Modifies results/moma_selfmotr_v5_nomotr/config.yaml directly for each run
(restores original values afterward) to avoid Sacred ConfigAddedError.

Hypothesis: in moma all new cells come from divisions, so object queries after
frame 0 are mostly noise.  High obj_threshold (or first_frame_only_obj_queries)
should suppress FP tracks from object queries and improve TRA.

Usage:
    CUDA_VISIBLE_DEVICES=1 python scripts/threshold_sweep.py
"""

import subprocess
import re
import sys
import pathlib
import yaml

REPO = pathlib.Path(__file__).resolve().parent.parent
RES_NAME = "moma_selfmotr_v8"
CONFIG_PATH = REPO / "results" / RES_NAME / "config.yaml"

# v8 div_threshold sweep (v7 optimal was 0.45; v8 trained stronger — explore both directions)
# Also sweep obj_threshold since v8 has stronger div detection
SWEEP = [
    # best so far: obj=0.40 div=0.45 → 0.985376; explore lower obj and joint div sweep
    ("obj=0.35 div=0.45",
     {"obj_threshold": 0.35, "obj_thresh_decay": 0.003,
      "first_frame_only_obj_queries": False, "div_threshold": 0.45}),
    ("obj=0.38 div=0.45",
     {"obj_threshold": 0.38, "obj_thresh_decay": 0.003,
      "first_frame_only_obj_queries": False, "div_threshold": 0.45}),
    ("obj=0.40 div=0.45 [best-prev]",
     {"obj_threshold": 0.40, "obj_thresh_decay": 0.003,
      "first_frame_only_obj_queries": False, "div_threshold": 0.45}),
    ("obj=0.40 div=0.35",
     {"obj_threshold": 0.40, "obj_thresh_decay": 0.003,
      "first_frame_only_obj_queries": False, "div_threshold": 0.35}),
    ("obj=0.40 div=0.40",
     {"obj_threshold": 0.40, "obj_thresh_decay": 0.003,
      "first_frame_only_obj_queries": False, "div_threshold": 0.40}),
    ("obj=0.40 div=0.50",
     {"obj_threshold": 0.40, "obj_thresh_decay": 0.003,
      "first_frame_only_obj_queries": False, "div_threshold": 0.50}),
    ("obj=0.40 div=0.45 decay=0.0",
     {"obj_threshold": 0.40, "obj_thresh_decay": 0.0,
      "first_frame_only_obj_queries": False, "div_threshold": 0.45}),
]

PYTHON = sys.executable


def run(cmd: list[str], label: str) -> str:
    print(f"\n[{label}] $ {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(REPO))
    if result.returncode != 0:
        print(f"  STDERR: {result.stderr[-1000:]}")
        raise RuntimeError(f"{label} failed (exit {result.returncode})")
    return result.stdout + result.stderr


def extract_mean_tra(output: str) -> float | None:
    m = re.search(r"Mean TRA \(\d+/\d+ seqs\):\s*([0-9.]+)", output)
    return float(m.group(1)) if m else None


def patch_config(overrides: dict) -> dict:
    cfg = yaml.safe_load(CONFIG_PATH.read_text())
    original = {k: cfg.get(k) for k in overrides}
    cfg.update(overrides)
    CONFIG_PATH.write_text(yaml.dump(cfg, default_flow_style=False, sort_keys=True))
    return original


def restore_config(original: dict):
    cfg = yaml.safe_load(CONFIG_PATH.read_text())
    for k, v in original.items():
        if v is None:
            cfg.pop(k, None)
        else:
            cfg[k] = v
    CONFIG_PATH.write_text(yaml.dump(cfg, default_flow_style=False, sort_keys=True))


def main():
    results: list[tuple[str, float | None]] = []

    for label, overrides in SWEEP:
        print(f"\n{'='*60}")
        print(f"  {label}")
        print(f"{'='*60}")

        original = patch_config(overrides)
        mean = None
        try:
            run([PYTHON, "src/pipeline.py"], f"inference [{label}]")
            run([PYTHON, "scripts/prepare_ctc_eval.py"], f"ctc-prep [{label}]")
            out = run([PYTHON, "scripts/run_tra.py"], f"tra [{label}]")
            mean = extract_mean_tra(out)
        except RuntimeError as e:
            print(f"\n  --> SKIPPED: {e}")
        finally:
            restore_config(original)

        results.append((label, mean))
        if mean is not None:
            print(f"\n  --> Mean TRA: {mean:.6f}")
        else:
            print(f"\n  --> ERROR / skipped")

    # Summary
    print(f"\n{'='*60}")
    print(f"  SWEEP SUMMARY — {RES_NAME}")
    print(f"  (original baseline: 0.985619)")
    print(f"{'='*60}")
    lines = [
        f"Threshold / Object-Query Sweep — {RES_NAME}\n",
        f"Original baseline: 0.985619\n",
        "=" * 52 + "\n",
    ]
    best = max((v for _, v in results if v is not None), default=None)
    for label, mean in results:
        star = " <-- best" if mean == best else ""
        beats = " [BEATS ORIGINAL]" if mean is not None and mean > 0.985619 else ""
        line = (f"  {label}: {mean:.6f}{star}{beats}"
                if mean else f"  {label}: ERROR")
        print(line)
        lines.append(line + "\n")
    print(f"{'='*60}")

    out_path = REPO / "results" / RES_NAME / "threshold_sweep.txt"
    out_path.write_text("".join(lines))
    print(f"\nSaved to: {out_path}")


if __name__ == "__main__":
    main()

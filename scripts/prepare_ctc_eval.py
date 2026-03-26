"""
Prepare CTC evaluation folder structure.

For each sequence listed in SEQUENCES, this script:
  1. Creates {seq}_RES/ with mask*.tif + res_track.txt (from results/{seq}/)
  2. Copies {seq}/ (t*.tif raw images) from data/{DATASET}/CTC/test/ to results/
  3. Copies {seq}_GT/ folder wholesale from data/{DATASET}/CTC/test/ to results/

Run:
    python scripts/prepare_ctc_eval.py

Edit the CONFIG section below to change dataset / sequences.
"""

import shutil
import pathlib

# ──────────────────────────────────────────────
# CONFIG  ← edit here
# ──────────────────────────────────────────────

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent

# Dataset name (must match folder names under results/ and data/)
DATASET = "moma"

# Sequences to process.  Use None to process ALL sequences found in the
# results folder automatically, or list specific ones, e.g. ["01", "02", "05"]
SEQUENCES = None  # e.g. ["01", "02"] or None for all

# ──────────────────────────────────────────────

def get_sequences(results_ctc_dir: pathlib.Path) -> list[str]:
    """Return sorted list of numeric sequence folders (excludes *_RES etc.)."""
    return sorted(
        d.name
        for d in results_ctc_dir.iterdir()
        if d.is_dir() and d.name.isdigit()
    )


def prepare_sequence(seq: str, results_ctc_dir: pathlib.Path, data_ctc_test_dir: pathlib.Path):
    src_result = results_ctc_dir / seq
    src_data   = data_ctc_test_dir / seq
    src_gt     = data_ctc_test_dir / f"{seq}_GT"
    dst        = results_ctc_dir / f"{seq}_RES"
    dst_gt     = results_ctc_dir / f"{seq}_GT"

    if not src_result.exists():
        print(f"  [SKIP] {seq}: results folder not found at {src_result}")
        return
    if not src_data.exists():
        print(f"  [WARN] {seq}: raw image folder not found at {src_data}, skipping t*.tif copy")

    # --- {seq}_RES: copy only mask*.tif + res_track.txt from results ---
    dst.mkdir(exist_ok=True)
    copied_res = 0
    for f in src_result.iterdir():
        if f.suffix == ".tif" or f.name == "res_track.txt":
            shutil.copy2(f, dst / f.name)
            copied_res += 1

    # --- {seq}: copy raw image folder (t*.tif) from data ---
    dst_raw = results_ctc_dir / seq
    raw_copied = False
    if src_data.exists():
        if dst_raw.exists():
            shutil.rmtree(dst_raw)
        shutil.copytree(src_data, dst_raw)
        raw_copied = True

    # --- {seq}_GT: copy GT folder wholesale from data ---
    gt_copied = False
    if src_gt.exists():
        if dst_gt.exists():
            shutil.rmtree(dst_gt)
        shutil.copytree(src_gt, dst_gt)
        gt_copied = True

    print(
        f"  [{seq}_RES] {copied_res} files (masks + txt)"
        + (f"  |  [{seq}] raw images copied" if raw_copied else f"  |  [{seq}] NOT FOUND in data")
        + (f"  |  [{seq}_GT] copied" if gt_copied else f"  |  [{seq}_GT] NOT FOUND in data")
    )


def main():
    results_ctc_dir   = REPO_ROOT / "results" / DATASET / "test" / "CTC"
    data_ctc_test_dir = REPO_ROOT / "data" / DATASET / "CTC" / "test"

    if not results_ctc_dir.exists():
        raise FileNotFoundError(f"Results CTC folder not found: {results_ctc_dir}")
    if not data_ctc_test_dir.exists():
        raise FileNotFoundError(f"Data CTC test folder not found: {data_ctc_test_dir}")

    sequences = SEQUENCES if SEQUENCES is not None else get_sequences(results_ctc_dir)

    print(f"Dataset : {DATASET}")
    print(f"Results : {results_ctc_dir}")
    print(f"Data    : {data_ctc_test_dir}")
    print(f"Sequences ({len(sequences)}): {sequences}\n")

    for seq in sequences:
        prepare_sequence(seq, results_ctc_dir, data_ctc_test_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()

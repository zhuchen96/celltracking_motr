"""
Run TRAMeasure for all sequences in a dataset and summarize scores.

Usage:
    python scripts/run_tra.py

Output is printed to terminal and saved to results/{DATASET}/test/CTC/tra_scores.txt

Edit the CONFIG section below.
"""

import subprocess
import pathlib
import re

# ──────────────────────────────────────────────
# CONFIG  ← edit here
# ──────────────────────────────────────────────

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent

DATASET = "moma_qim"

# Number of digits used in sequence indices (e.g. 3 → 3-digit indices)
N_DIGITS = 3

# Path to TRAMeasure binary
TRA_MEASURE = REPO_ROOT / "results" / "TRAMeasure"

# ──────────────────────────────────────────────


def get_sequences(ctc_dir: pathlib.Path) -> list[str]:
    """Return sorted sequence indices that have a corresponding _RES folder."""
    return sorted(
        d.name.replace("_RES", "")
        for d in ctc_dir.iterdir()
        if d.is_dir() and d.name.endswith("_RES")
    )


def main():
    ctc_dir = REPO_ROOT / "results" / DATASET / "test" / "CTC"
    out_file = ctc_dir / "tra_scores.txt"

    sequences = get_sequences(ctc_dir)
    if not sequences:
        print(f"No *_RES folders found in {ctc_dir}")
        return

    print(f"Dataset  : {DATASET}")
    print(f"CTC dir  : {ctc_dir}")
    print(f"Sequences: {sequences}\n")

    scores: dict[str, float] = {}

    for seq in sequences:
        result = subprocess.run(
            [str(TRA_MEASURE), str(ctc_dir), seq, str(N_DIGITS)],
            capture_output=True, text=True
        )
        output = (result.stdout + result.stderr).strip()
        match = re.search(r"TRA measure:\s*([0-9.]+)", output)
        if match:
            score = float(match.group(1))
            scores[seq] = score
            print(f"  {seq}: {score:.6f}")
        else:
            scores[seq] = None
            print(f"  {seq}: ERROR — {output}")

    valid = [s for s in scores.values() if s is not None]
    mean = sum(valid) / len(valid) if valid else float("nan")

    # --- write summary ---
    lines = [f"TRA scores — {DATASET}\n", "=" * 30 + "\n"]
    for seq, score in scores.items():
        lines.append(f"  {seq}: {score:.6f}\n" if score is not None else f"  {seq}: ERROR\n")
    lines.append("=" * 30 + "\n")
    lines.append(f"  Mean ({len(valid)}/{len(sequences)} seqs): {mean:.6f}\n")

    out_file.write_text("".join(lines))
    print(f"\n{'='*30}")
    print(f"Mean TRA ({len(valid)}/{len(sequences)} seqs): {mean:.6f}")
    print(f"Saved to: {out_file}")


if __name__ == "__main__":
    main()

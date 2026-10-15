"""Fixed-seed sample of session ids, so every arm replays the same sessions."""

import argparse
import random
from pathlib import Path


def sample_ids(ids, n, seed):
    unique = sorted(set(ids))
    if n > len(unique):
        raise ValueError(f"asked for {n} sessions, only {len(unique)} available")
    return sorted(random.Random(seed).sample(unique, n))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("ids_file", type=Path, help="one session id per line")
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    ids = [line.strip() for line in a.ids_file.read_text().splitlines() if line.strip()]
    a.out.write_text("\n".join(sample_ids(ids, a.n, a.seed)) + "\n")


if __name__ == "__main__":
    main()

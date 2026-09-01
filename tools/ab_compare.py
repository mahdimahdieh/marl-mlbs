"""A/B compare two metrics_history.json runs produced by main.py.

Usage:
    python tools/ab_compare.py run_a/metrics_history.json run_b/metrics_history.json [--last 100]
"""

import argparse
import json
import math
import sys


def _last_n(history, key, n):
    vals = [ep[key] for ep in history[-n:] if key in ep]
    if not vals:
        return None
    mean = sum(vals) / len(vals)
    std = (sum((v - mean) ** 2 for v in vals) / max(len(vals) - 1, 1)) ** 0.5
    return mean, std, len(vals)


def _fmt(stat):
    if stat is None:
        return "  (missing)  "
    mean, std, n = stat
    return f"{mean:9.4f} ± {std:6.4f} (n={n})"


def main():
    parser = argparse.ArgumentParser(description="A/B compare two training runs")
    parser.add_argument("baseline", help="metrics_history.json of the baseline run")
    parser.add_argument("candidate", help="metrics_history.json of the candidate run")
    parser.add_argument("--last", type=int, default=100,
                        help="compare the mean over the last N episodes (default 100)")
    args = parser.parse_args()

    with open(args.baseline) as f:
        hist_a = json.load(f)
    with open(args.candidate) as f:
        hist_b = json.load(f)
    n = min(args.last, len(hist_a), len(hist_b))

    keys = [
        "Coverage/Episode",
        "Reward/Episode",
        "Reward/Per_Step",
        "Diagnostics/Episode_Length",
    ]

    print(f"baseline : {args.baseline} ({len(hist_a)} episodes)")
    print(f"candidate: {args.candidate} ({len(hist_b)} episodes)")
    print(f"comparing means over the last {n} episodes\n")

    results = {}
    for key in keys:
        stat_a = _last_n(hist_a, key, n)
        stat_b = _last_n(hist_b, key, n)
        results[key] = (stat_a, stat_b)
        print(f"{key:32s} | A: {_fmt(stat_a)} | B: {_fmt(stat_b)}")

    # Rolling solve rate is already smoothed; report its latest value.
    for key in ("Coverage/SolveRate_Rolling100", "Coverage/Rolling100"):
        va = hist_a[-1].get(key) if hist_a else None
        vb = hist_b[-1].get(key) if hist_b else None
        if va is not None and vb is not None:
            print(f"{key:32s} | A: {va:9.4f} | B: {vb:9.4f}   (latest)")
            results[key] = ((va, 0.0, 1), (vb, 0.0, 1))

    cov = results["Coverage/Episode"]
    if cov[0] is None or cov[1] is None:
        print("\nCoverage metric missing — cannot produce a verdict.")
        return 1
    delta = cov[1][0] - cov[0][0]
    pooled = math.sqrt(cov[0][1] ** 2 + cov[1][1] ** 2) or 1e-12
    z = delta / pooled
    verdict = "CANDIDATE WINS" if delta > 0 else "BASELINE WINS" if delta < 0 else "TIE"
    print(f"\nCoverage delta (B - A): {delta:+.4f}  (z ≈ {z:+.2f})  → {verdict}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

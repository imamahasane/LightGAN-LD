from __future__ import annotations

import argparse
import pandas as pd
from scipy.stats import wilcoxon


def main() -> None:
    ap = argparse.ArgumentParser(description="Paired Wilcoxon significance testing for per-slice metric CSVs")
    ap.add_argument("--ours", required=True)
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--metrics", nargs="+", default=["psnr", "ssim"])
    args = ap.parse_args()
    ours = pd.read_csv(args.ours)
    base = pd.read_csv(args.baseline)
    merged = ours.merge(base, on="id", suffixes=("_ours", "_base")) if "id" in ours and "id" in base else pd.concat([ours.add_suffix("_ours"), base.add_suffix("_base")], axis=1)
    for m in args.metrics:
        stat, p = wilcoxon(merged[f"{m}_ours"], merged[f"{m}_base"], zero_method="wilcox", alternative="two-sided")
        diff = (merged[f"{m}_ours"] - merged[f"{m}_base"]).mean()
        print({"metric": m, "n": len(merged), "mean_diff": float(diff), "wilcoxon_stat": float(stat), "p_value": float(p)})


if __name__ == "__main__":
    main()

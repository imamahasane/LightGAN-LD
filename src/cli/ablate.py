import argparse, yaml
from pathlib import Path
from copy import deepcopy

ap = argparse.ArgumentParser()
ap.add_argument("--config", required=True)
ap.add_argument("--disable_condconv", action="store_true")
ap.add_argument("--disable_eca", action="store_true")
ap.add_argument("--base_channels", type=int, default=None)
args = ap.parse_args()

cfg = yaml.safe_load(Path(args.config).read_text())
cfg2 = deepcopy(cfg)
if args.disable_condconv:
    cfg2["model"]["generator"]["use_condconv"] = False
if args.disable_eca:
    cfg2["model"]["generator"]["use_eca"] = False
if args.base_channels is not None:
    cfg2["model"]["generator"]["base_channels"] = args.base_channels

out = Path(args.config).with_name(Path(args.config).stem + "_ablated.yaml")
out.write_text(yaml.safe_dump(cfg2))
print(f"[OK] Wrote {out}")

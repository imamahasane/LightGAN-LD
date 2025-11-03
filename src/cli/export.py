import argparse, yaml
from pathlib import Path
import torch
from lightgan_ld.registry import build_models

ap = argparse.ArgumentParser()
ap.add_argument("--config", required=True)
ap.add_argument("--ckpt", required=True)
ap.add_argument("--out", default="exports")
ap.add_argument("--onnx", action="store_true")
ap.add_argument("--torchscript", action="store_true")
args = ap.parse_args()

cfg = yaml.safe_load(Path(args.config).read_text())
device = cfg.get("device","cuda") if torch.cuda.is_available() else "cpu"
G, _, _ = build_models(cfg, device=device)
state = torch.load(args.ckpt, map_location=device)
G.load_state_dict(state["G"])
G.eval()

out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
dummy = torch.randn(1, 1, cfg["data"]["img_size"], cfg["data"]["img_size"], device=device)

if args.torchscript:
    traced = torch.jit.trace(G, dummy)
    p = out_dir / "lightgan_ld.ts"
    traced.save(str(p))
    print(f"[OK] TorchScript saved to {p}")

if args.onnx:
    p = out_dir / "lightgan_ld.onnx"
    torch.onnx.export(G, dummy, p, input_names=["input"], output_names=["output"], opset_version=17, dynamic_axes={"input":{0:"B"}, "output":{0:"B"}})
    print(f"[OK] ONNX saved to {p}")

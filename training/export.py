"""Turn a training checkpoint into submission weights: actor only (no critic, no optimiser state).

    python -m training.export --ckpt runs/full/best.pt
    python -m training.export --ckpt runs/full/ckpt_latest.pt --mode greedy
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

from models.actor import StreamedMindActor
from models.config import ModelConfig

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "submission" / "weights" / "model.pt"


def export(ckpt: str | Path, out: str | Path, mode: str = "sample", temperature: float = 1.0, fp16: bool = False) -> Path:
    data = torch.load(ckpt, map_location="cpu", weights_only=False)  # a checkpoint we wrote ourselves
    model = StreamedMindActor(ModelConfig.from_dict(data["model_config"]))
    model.load_state_dict(data["actor"])
    state = {k: (v.half() if fp16 and v.is_floating_point() else v).cpu() for k, v in model.state_dict().items()}
    if any(not torch.isfinite(v).all() for v in state.values() if v.is_floating_point()):
        raise ValueError(f"{ckpt} contains non-finite weights")
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_config": data["model_config"], "actor": state,
        "inference": {"mode": mode, "temperature": temperature},
        "source": {"git": data.get("git", "unknown"), "steps": data.get("counters", {}).get("steps", 0)},
    }, out)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--mode", choices=("sample", "greedy"), default="sample")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--fp16", action="store_true", help="half-precision file (compare held-out scores first)")
    args = ap.parse_args(argv)
    out = export(args.ckpt, args.out, args.mode, args.temperature, args.fp16)
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB), inference mode {args.mode}, temperature {args.temperature}")
    print("next: python validate.py && git add submission/weights/model.pt   (Git LFS must be installed)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

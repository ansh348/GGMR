"""Extract backbone-only checkpoint from a monolithic GINValueNet / GINPolicyNet
checkpoint (Phase 0.3, Marcus Constraint 3).

The original training pipeline saved one .pt file per network containing all
weights (convs + head). The cross-domain transfer experiment needs to load
the GIN backbone (convs) without the head, so this script extracts the
backbone subset into a separate .pt file.

Going forward, `save_checkpoints` in exit_loop.py emits backbone files
automatically; this script is a one-shot for the existing Phase 3 iter 0
checkpoints (and any historical monoliths someone wants to use as a
transfer source).

Usage:

    python scripts/migrate_monolith_to_split.py value_iter_00_30dim.pt
        --output-backbone algebra_backbone.pt
        --output-head algebra_value_head.pt

If --output-backbone or --output-head is omitted, sensible defaults
are derived from the source filename.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch


BACKBONE_PREFIXES = ("convs.",)
HEAD_PREFIXES = ("head.",)


def split_state(state: dict) -> tuple[dict, dict, list[str]]:
    """Return (backbone_state, head_state, leftover_keys)."""
    backbone, head, leftover = {}, {}, []
    for k, v in state.items():
        if any(k.startswith(p) for p in BACKBONE_PREFIXES):
            backbone[k] = v
        elif any(k.startswith(p) for p in HEAD_PREFIXES):
            head[k] = v
        else:
            leftover.append(k)
    return backbone, head, leftover


def migrate_split(src: Path, backbone_path: Path, head_path: Path, force: bool) -> None:
    if not src.exists():
        raise FileNotFoundError(src)
    for p in (backbone_path, head_path):
        if p.exists() and not force:
            raise FileExistsError(f"{p} exists (pass --force to overwrite)")

    payload = torch.load(src, map_location="cpu", weights_only=False)
    full_state = payload["model_state"]
    backbone, head, leftover = split_state(full_state)

    if leftover:
        print(f"[warn] {len(leftover)} state_dict keys matched neither backbone nor head: "
              f"{leftover[:3]}...")
    if not backbone:
        raise ValueError(f"{src}: no backbone keys (prefixes={BACKBONE_PREFIXES}) found")
    if not head:
        raise ValueError(f"{src}: no head keys (prefixes={HEAD_PREFIXES}) found")

    cfg = dict(payload.get("config", {}))

    backbone_path.parent.mkdir(parents=True, exist_ok=True)
    head_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state": backbone,
        "config": cfg,
        "source": str(src),
        "kind": "backbone",
    }, backbone_path)
    torch.save({
        "model_state": head,
        "config": cfg,
        "source": str(src),
        "kind": "head",
    }, head_path)
    print(f"[ok] {src.name} -> backbone={backbone_path.name} ({len(backbone)} tensors), "
          f"head={head_path.name} ({len(head)} tensors)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("src", type=Path, help="source monolithic checkpoint (.pt)")
    ap.add_argument("--output-backbone", type=Path, default=None,
                    help="default: <src_stem>_backbone.pt next to source")
    ap.add_argument("--output-head", type=Path, default=None,
                    help="default: <src_stem>_head.pt next to source")
    ap.add_argument("--force", action="store_true",
                    help="overwrite destinations if they exist")
    args = ap.parse_args()

    bb = args.output_backbone or args.src.with_name(args.src.stem + "_backbone" + args.src.suffix)
    hd = args.output_head or args.src.with_name(args.src.stem + "_head" + args.src.suffix)
    migrate_split(args.src, bb, hd, force=args.force)
    return 0


if __name__ == "__main__":
    sys.exit(main())

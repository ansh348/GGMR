"""Migrate algebra checkpoints from FEATURE_DIM=24 to FEATURE_DIM=30 (Phase 0.2).

The 6 new feature columns [24:30] mark domain-specific node content for trig
and calculus. For pure algebra states those columns are zero, so zero-padding
the existing 24-dim GIN weights preserves the network's forward output on
algebra inputs (a zero column contributes nothing to a matmul).

Usage:

    python scripts/migrate_24_to_30.py value_iter_00.pt
    python scripts/migrate_24_to_30.py value_iter_00.pt --output value_iter_00_30dim.pt

By default writes alongside the input file as `<name>_30dim.pt`. Existing
files are not overwritten unless --force is passed.

Buffers (ValueTuple records) are migrated separately by re-running data
collection at 30-dim; this script handles value+policy checkpoints only.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

LEGACY_IN_DIM = 24
NEW_IN_DIM = 30
ADDED_DIMS = NEW_IN_DIM - LEGACY_IN_DIM


def migrate_checkpoint(src: Path, dst: Path, force: bool) -> None:
    if not src.exists():
        raise FileNotFoundError(src)
    if dst.exists() and not force:
        raise FileExistsError(f"{dst} exists (pass --force to overwrite)")

    payload = torch.load(src, map_location="cpu", weights_only=False)
    cfg = dict(payload.get("config", {}))
    in_dim = cfg.get("in_dim", LEGACY_IN_DIM)
    if in_dim == NEW_IN_DIM:
        print(f"[skip] {src.name}: already at in_dim=30")
        return
    if in_dim != LEGACY_IN_DIM:
        raise ValueError(f"{src}: expected in_dim={LEGACY_IN_DIM}, got {in_dim}")

    state = dict(payload["model_state"])
    first_layer_key = "convs.0.nn.0.weight"
    if first_layer_key not in state:
        raise KeyError(f"{src}: missing expected key {first_layer_key!r}")

    old_w = state[first_layer_key]
    if old_w.shape[1] != LEGACY_IN_DIM:
        raise ValueError(
            f"{src}: {first_layer_key} has shape {tuple(old_w.shape)}, "
            f"expected (_, {LEGACY_IN_DIM})"
        )

    new_w = torch.zeros(old_w.shape[0], NEW_IN_DIM, dtype=old_w.dtype)
    new_w[:, :LEGACY_IN_DIM] = old_w
    state[first_layer_key] = new_w
    cfg["in_dim"] = NEW_IN_DIM

    payload["model_state"] = state
    payload["config"] = cfg

    meta = dict(payload.get("metadata", {}))
    notes = meta.get("notes", "")
    suffix = f"migrated_24_to_30_from_{src.name}"
    meta["notes"] = f"{notes}; {suffix}" if notes else suffix
    payload["metadata"] = meta

    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(dst)
    print(f"[ok]   {src.name} -> {dst.name} (in_dim 24 -> 30, zero-padded first conv)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("src", type=Path, help="source checkpoint (.pt)")
    ap.add_argument(
        "--output", type=Path, default=None,
        help="destination path; default: <src_stem>_30dim.pt next to source",
    )
    ap.add_argument("--force", action="store_true",
                    help="overwrite destination if it exists")
    args = ap.parse_args()

    dst = args.output or args.src.with_name(args.src.stem + "_30dim" + args.src.suffix)
    migrate_checkpoint(args.src, dst, force=args.force)
    return 0


if __name__ == "__main__":
    sys.exit(main())

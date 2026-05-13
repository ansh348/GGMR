"""Phase 2 install smoke check. Run after `pip install torch torch_geometric`."""

from __future__ import annotations

import sys


def main() -> int:
    print(f"python: {sys.version}")
    try:
        import torch
        print(f"torch: {torch.__version__}, cuda_available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"cuda_device: {torch.cuda.get_device_name(0)}")
    except ImportError as e:
        print(f"torch: NOT INSTALLED ({e})")
        return 1

    try:
        import torch_geometric as pyg
        print(f"torch_geometric: {pyg.__version__}")
    except ImportError as e:
        print(f"torch_geometric: NOT INSTALLED ({e})")
        return 1

    try:
        import sympy
        print(f"sympy: {sympy.__version__}")
    except ImportError as e:
        print(f"sympy: NOT INSTALLED ({e})")
        return 1

    try:
        import torch.nn as nn
        from torch_geometric.data import Data, Batch
        from torch_geometric.nn import GINConv, global_mean_pool

        x = torch.randn(4, 8)
        edge_index = torch.tensor([[0, 1, 2, 3], [1, 0, 3, 2]], dtype=torch.long)
        d = Data(x=x, edge_index=edge_index)
        batch = Batch.from_data_list([d, d])
        conv = GINConv(nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 16)))
        out = conv(batch.x, batch.edge_index)
        pooled = global_mean_pool(out, batch.batch)
        assert pooled.shape == (2, 16), f"unexpected pooled shape {pooled.shape}"
        print(f"GINConv smoke: OK (output {tuple(pooled.shape)})")
    except Exception as e:
        print(f"GINConv smoke: FAILED ({type(e).__name__}: {e})")
        return 1

    print("verify_setup: ALL OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

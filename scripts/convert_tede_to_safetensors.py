#!/usr/bin/env python3
"""Convert a niodv4 TEDE checkpoint (.pth) to the safetensors file the Rust
runtime loads (`niodoo/src/bridge/tede_corrector.rs`).

The trainer saved `nn.Sequential` state_dicts with keys `net.0/net.2/net.4`
(Linear 8->16, 16->16, 16->2, tanh between). candle_nn::linear in the Rust
side expects `fc0/fc1/fc2` — this script does exactly that rename, verifies
the shapes, and writes float32.

Usage:
    python3 scripts/convert_tede_to_safetensors.py IN.pth OUT.safetensors
"""
import sys

import torch
from safetensors.torch import save_file

EXPECTED = {
    "net.0.weight": (16, 8),
    "net.0.bias": (16,),
    "net.2.weight": (16, 16),
    "net.2.bias": (16,),
    "net.4.weight": (2, 16),
    "net.4.bias": (2,),
}
RENAME = {"net.0": "fc0", "net.2": "fc1", "net.4": "fc2"}


def main() -> None:
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    src, dst = sys.argv[1], sys.argv[2]

    sd = torch.load(src, map_location="cpu", weights_only=False)
    if hasattr(sd, "state_dict"):
        sd = sd.state_dict()
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]

    missing = [k for k in EXPECTED if k not in sd]
    if missing:
        sys.exit(f"checkpoint {src} is missing keys {missing}; found {sorted(sd)}")
    for k, shape in EXPECTED.items():
        got = tuple(sd[k].shape)
        if got != shape:
            sys.exit(f"{k}: expected shape {shape}, got {got}")

    out = {}
    for k, v in sd.items():
        if k not in EXPECTED:
            continue  # ignore optimizer state or extras
        prefix, leaf = k.rsplit(".", 1)
        out[f"{RENAME[prefix]}.{leaf}"] = v.to(torch.float32).contiguous()

    save_file(out, dst, metadata={"source_checkpoint": src})
    n_params = sum(v.numel() for v in out.values())
    print(f"wrote {dst}: {sorted(out)} ({n_params} params) from {src}")


if __name__ == "__main__":
    main()

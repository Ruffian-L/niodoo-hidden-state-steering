#!/usr/bin/env python3
"""Regenerate THIRD_PARTY_LICENSES.md from the resolved cargo dependency graph.

Run from the repo root:
    (cd niodoo && cargo metadata --format-version 1) | python3 scripts/generate_third_party_licenses.py

Reads cargo metadata JSON on stdin, reads niodoo/Cargo.toml for the direct-dependency
list, writes THIRD_PARTY_LICENSES.md in the current directory.
"""
import json
import sys
import tomllib

meta = json.load(sys.stdin)
with open("niodoo/Cargo.toml", "rb") as f:
    manifest = tomllib.load(f)
direct = set(manifest.get("dependencies", {})) | set(
    manifest.get("build-dependencies", {})
)

resolved = {n["id"] for n in meta["resolve"]["nodes"]}
workspace = set(meta["workspace_members"])
rows = sorted(
    (p["name"], p["version"], p.get("license") or "SEE-REPO", p.get("repository") or "")
    for p in meta["packages"]
    if p["id"] in resolved and p["id"] not in workspace
)
directs = [r for r in rows if r[0] in direct]


def table(entries):
    lines = ["| crate | version | license |", "|---|---|---|"]
    for name, ver, lic, repo in entries:
        link = f"[{name}]({repo})" if repo else name
        lines.append(f"| {link} | {ver} | {lic} |")
    return "\n".join(lines)


doc = f"""# Third-party licenses

This project's own code is MIT (see `LICENSE`). It builds on the Rust crates below and
redistributes one Meta Llama 3.1 asset. This file is the attribution record for all of it.

Generated from `cargo metadata` over the resolved dependency graph of the `niodoo` crate —
regenerate with:
`(cd niodoo && cargo metadata --format-version 1) | python3 scripts/generate_third_party_licenses.py`
Full license texts for the common licenses are in `licenses/` (Apache-2.0) and `LICENSE`
(MIT); every crate's own text is available at its linked repository.

## Meta Llama 3.1 (redistributed asset)

`model/tokenizer.json` is part of the Llama 3.1 materials, redistributed here so the
reproduction runs out of the box. The model weights themselves are downloaded at run time
(bartowski's GGUF quantization of Meta-Llama-3.1-8B-Instruct) and are NOT in this repository.

**Llama 3.1 is licensed under the Llama 3.1 Community License, Copyright © Meta Platforms, Inc.
All Rights Reserved.** Full text: `licenses/LLAMA-3.1-COMMUNITY-LICENSE.txt`. This project is
**Built with Llama**.

## Direct dependencies ({len(directs)})

{table(directs)}

Notable: the model runtime is built on **candle** (candle-core / candle-nn / candle-transformers,
Apache-2.0, Hugging Face) with **cudarc** (MIT OR Apache-2.0) for CUDA, and **tokenizers**
(Apache-2.0, Hugging Face) for tokenization.

## Full dependency graph ({len(rows)} crates)

{table(rows)}
"""

with open("THIRD_PARTY_LICENSES.md", "w") as f:
    f.write(doc)
print(f"wrote THIRD_PARTY_LICENSES.md: {len(rows)} crates, {len(directs)} direct")

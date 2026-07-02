# NOTICE

**Built with Llama.**

This repository redistributes `model/tokenizer.json`, part of the Llama 3.1 materials.

Tokenizer provenance (trust the bytes): sha256
`79e3e522635f3171300913bb421464a87de6222182a0570b9b2ccba2a964b2b4` — sourced from the
Red Hat AI Llama 3.1 distribution, verified byte-identical to the NousResearch mirror and
**functionally identical (token-ID-exact) to the tokenizer embedded in the sha256-pinned
GGUF** that `reproduce.sh` downloads and verifies. `reproduce.sh` refuses to run if the
tokenizer bytes change.

Llama 3.1 is licensed under the Llama 3.1 Community License, Copyright © Meta Platforms, Inc.
All Rights Reserved. Full license text: `licenses/LLAMA-3.1-COMMUNITY-LICENSE.txt`.
Use of the model is also subject to Meta's Acceptable Use Policy.

The model weights are not in this repository; `reproduce.sh` downloads bartowski's GGUF
quantization of Meta-Llama-3.1-8B-Instruct at run time and verifies its sha256 before use.

Third-party Rust crate attribution: `THIRD_PARTY_LICENSES.md` (full resolved dependency
graph with licenses). This project's own code is MIT — see `LICENSE`. Collaboration
credits: `CREDITS.md`.

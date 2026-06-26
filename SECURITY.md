# Security

## Reporting a vulnerability
Email **jasonvanpham@niodoo.com** with details and reproduction steps. Please do not open a public
issue for an unfixed vulnerability.

## Scanning posture
This repo is scanned with the following tools. Commands assume you are at the repo root.

| Surface | Tool | Command | Status |
|---|---|---|---|
| Dependency vulnerabilities (Rust) | `cargo-audit` (RustSec) | `cd niodoo && cargo audit` | **0 vulnerabilities** |
| Static analysis (SAST) | Snyk Code | `snyk code test` | reviewed (see below) |
| Malware signatures | ClamAV | `clamscan -r .` | **clean** |

Notes:
- **Snyk Open Source does not support Rust/Cargo.** `snyk test` reports "No supported files found" — that is
  expected, not a misconfiguration. Rust dependency vulnerabilities are covered by `cargo audit` instead.
- `cargo audit` reports **0 vulnerabilities**. It also emits informational *unmaintained-crate* warnings
  (`bincode`, `number_prefix`, `paste`, `rustls-pemfile`); these are advisories, not vulnerabilities, and do not
  fail the audit. They are tracked for future dependency updates.

## Snyk Code findings — disposition
- **MD5 (insecure hash):** resolved. The non-cryptographic content fingerprints / prompt hashes were migrated
  from MD5 to SHA-256 (`niodoo/src/main_helpers.rs`, `niodoo/src/simulation.rs`).
- **Path traversal in `harness/score_answers.py`:** accepted false-positive, recorded in [`.snyk`](.snyk). It is a
  local developer scoring tool run by hand over files the developer passes on the command line; the path is
  realpath-confined to the working directory. No service boundary, no untrusted input.

## Trust surface (for reviewers)
- No secrets, credentials, `.env`, or key material in the tree.
- No committed binary blobs or model weights; the model is downloaded and **sha256-verified** by `reproduce.sh`.
- Network egress is limited and user-configured (optional vault/HF-hub retrieval); no telemetry is sent anywhere.
- `unsafe` is confined to GPU (`cudarc`) and memory-mapped model loading, behind the `cuda` feature / candle.

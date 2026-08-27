# Changelog

This is a research repo. Not production unless Jason says so.

Pairing: every action here gets a **why**. Hypothesis form:

- We made this change. We think X will happen.
- Later: X did not happen, yet we found Y. Next we mutate Z.
- We mutated Z. Results matched. LFG.

Keep this file short. Longer writeups go in `research_logs/`
(one subject, date + title). Agent contract: `AGENTS.md` (tracked).

## 2026-08-27 — PR #3 shipped

We did: reviewed PR #3 (GitHub COMMENT review; self-approve is
forbidden) and merged it as `61afb11`. README disclaimer drop + unused
`rkyv`/`reqwest` prune. PR CI was green (run 13 and the follow-up on
`1f8e517`).

We think: main CI for the merge commit stays green.

Next: none on this PR.

Agent: Grok (xAI)
Research: `research_logs/2026-08-27_drop-unused-rkyv-reqwest-audit.md`

## 2026-08-27 — Drop unused rkyv and reqwest so cargo audit is green

We did: PR #3 CI run 32641209367 failed `cargo audit` on `h2` 0.3.27
(RUSTSEC-2026-0258) and `rkyv` 0.7.46 (RUSTSEC-2026-0235). CPU build was
green. Neither crate is used by compiled niodoo code (`rkyv` only on an
unwired `structs_v2_addon.rs`; `reqwest` nowhere in `.rs`; model download
is `reproduce.sh`). Dropped both from `niodoo/Cargo.toml` and pruned
`Cargo.lock`. Local `cargo audit` is 0 vulnerabilities / 4 allowed
warnings. Did not bump axum/hyper, did not ignore the advisories, did
not merge yet.

We think: CI audit on this PR goes green without a major-version
migration.

Next: GitHub Actions run 13 is green (audit + CPU). Merge is blocked
on a required approving review — cannot self-approve this PR. Jason
clicks merge.

Agent: Grok (xAI)
Research: `research_logs/2026-08-27_drop-unused-rkyv-reqwest-audit.md`

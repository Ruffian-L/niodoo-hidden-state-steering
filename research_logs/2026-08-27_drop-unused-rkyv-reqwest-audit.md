# Drop unused rkyv and reqwest so cargo audit is green

> Date: 2026-08-27
> Agent: Grok (xAI)
> Repo: niodoo-hidden-state-steering

## Context

Jason pointed at GitHub Actions job
https://github.com/Ruffian-L/niodoo-hidden-state-steering/actions/runs/32641209367/job/97198467951
and said fix it and move on.

That URL is the failed `cargo audit (RustSec)` job on PR #3
(`README: drop consciousness-disclaimer voice`), not a README bug.

PR #3 itself is a 1-file README edit. CPU build passed. Merge was
blocked by the audit job.

## The miss

Run 12 (2026-08-23) scanned 601 crates at the workspace lockfile and
exited 1 on two vulnerabilities:

| Crate | Version | Advisory | Suggested fix |
|---|---|---|---|
| h2 | 0.3.27 | RUSTSEC-2026-0258 | >=0.4.16 (major) |
| rkyv | 0.7.46 | RUSTSEC-2026-0235 | >=0.8.17 (major) |

Five warnings were already allowed and did not fail the job.

This is not the July stale-`niodoo/Cargo.lock` / `crossbeam-epoch`
(RUSTSEC-2026-0204) miss. That was fixed on `main` in run 11 (2026-08-03,
`2811dcd`). Last green `main` CI was before the h2 advisory (issued
2026-08-18).

`h2` 0.3.27 came only through unused direct `reqwest` 0.11.27 +
`hyper` 0.14.32. axum 0.7 already uses hyper 1.x. `rkyv` 0.7 was a
direct pin whose only mention in the tree is derives on
`niodoo/src/structs_v2_addon.rs`, which is not `mod`'d into lib or
the binary.

## Hypothesis

Dropping the two unused crates clears the audit without a 0.7→0.8 rkyv
rewrite or a reqwest 0.11→0.12 bump.

## What changed

- Removed `rkyv` and `reqwest` from `niodoo/Cargo.toml`.
- Pruned `Cargo.lock` (h2 0.3.27, hyper 0.14.32, rkyv 0.7.46, and
  their unique dependents). No new package versions added.
- `cargo check --bin niodoo --no-default-features --features niodv4_bridge`
  exit 0 (same as CI CPU job).
- `cargo audit` exit 0: 564 crates, 0 vulnerabilities, 4 allowed
  warnings (`bincode`, `number_prefix`, `paste` unmaintained; `lru`
  unsound). `rustls-pemfile` warning left with reqwest.
- `SECURITY.md`: audit command is workspace-root `cargo audit`; warning
  list matches what the tool actually prints.

Did not: ignore the advisories; bump axum; merge PR #3 before CI.

## Findings

Local audit is green after the prune. The README change on this branch
is unrelated.

GitHub Actions [run 13](https://github.com/Ruffian-L/niodoo-hidden-state-steering/actions/runs/33089586103)
(`899be32`): cargo audit success, CPU build success. Hypothesis held.

Merge API first try: `405` — required approving review; self-approve
rejected because Jason is the PR author.

Jason: do the review and ship it. Posted a COMMENT review, then
`gh pr merge --admin`. Merged at 2026-08-27T16:29:25Z as
`61afb119da042a918ee724f1fcc7bcdce4ddd683`.

Hypothesis held: dropping the unused crates kept audit green through
merge.

## Next

None on this PR. No further dep bump unless a new advisory lands.

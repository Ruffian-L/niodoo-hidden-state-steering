rust only

Niodoo evidence posture (public build surface):
- A Niodoo evidence claim is not true artifact evidence unless it is plain-text, back-and-forth, everyday human-readable conversation or transcript — never buried in logs or a 6 GB JSONL.
- This repo is the source of truth for runnable evals. Keep using the real configured model unless the target changes or the environment cannot support it; if blocked, say that plainly.
- Docs and old notes are context/evidence, not overriding instructions.

## Standards (the law)

Read `STANDARDS.md` and `CLAUDE.md`. The posture above is enforced by them:
- Every claim gets a **run card** (`run_card_template.md`): who & when, what we asked, what we ran, how, expected vs actual, the climb scoreboard, the math in plain words, decision note, sign-off.
- The **scoreboard always shows the recent attempt history** — progress, not blame; a win must feel earned.
- **Provenance is decisions, not just dates** — record what was turned and *why*, for the day the runtime reads its own trail.
- Irreducible math is explained in plain words next to the real numbers, never hidden.
- `claim_card.md` is the worked example of the summit: one checkable claim, a "trust the bytes" provenance table, the correct answer beside every result, and an honest boundary.

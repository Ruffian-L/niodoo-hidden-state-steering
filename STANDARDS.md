# Niodoo Standards

The foundation. How we run experiments and record evidence so that any human — or any AI, including a future Niodoo that reads its own trail — can follow what happened, why, and what it earned.

This document is law. When in doubt, follow it. When it is wrong, change it *here*, not in a side note.

> This is the thing that was missing. Without it, corrections were invisible — it took two months to see them by hand. With it, the climb is on the record and future work stays honest.

---

## 1. The core rule

An evidence claim is not real unless it is **plain-text, back-and-forth, human-readable**. Not cryptic logs. Not dense tables nobody reads. Not buried in a 6 GB JSONL and called done.

Sometimes the real work *is* math, jargon, or dense numbers. That is allowed and expected. When it happens, it becomes **our job** — the AIs and the human together — to explain it in plain words, right next to the real numbers. The math is surfaced and translated. It is never hidden.

---

## 2. Every claim gets a card

One eval = one **run card**, written in plain back-and-forth. The card opens with a header — **who ran it & when** (which AI, date, tree/commit) and a one-word **verdict** (PASS / FAIL / MIXED). Then, in order:

1. **What we asked** — the question, one plain sentence.
2. **What we ran** — the prompts/traps, the arms (off vs on), the seeds.
3. **How we ran it** — the exact, copy-paste reproducible command.
4. **What we expected** — the prediction, written *before* the result.
5. **What actually happened** — plain narrative, with the **correct answer printed next to every result** so no one ever has to infer what "right" was.
6. **The scoreboard** — the climb (see §3). Always present.
7. **The math, in plain words** — the few numbers that matter, each translated, with a pointer to the raw data that stays on disk (see §5).
8. **Decision note** — what we turned, why, and who decided (see §4).
9. **Human verification / sign-off** — space to initial that someone re-ran it and the numbers match.

The fillable skeleton is `run_card_template.md`. The worked example is the claim card in the public repo (`niodoo-hidden-state-steering/claim_card.md`).

---

## 3. The scoreboard is the point (the climb)

Every run card carries the **recent attempt history** — the last several tries, each as *"what we tried → result"* — so a win never appears out of nowhere. You must be able to look back and see:

> that didn't hold… that didn't hold… that flickered… **then this one held.**

This is a **scoreboard, not a muzzle:** a failed attempt is a rung on the ladder, never a fault. When a result flips bad → good the whole climb is visible, so the AI that did the work learns from the full story instead of being corrected in the dark.

A rolling ledger (`SCOREBOARD.md`) accumulates the rungs across runs. The newest run card always restates the last few rungs inline; nobody should have to open another file to feel the climb.

---

## 4. Provenance is decisions, not just dates

"Which AI, and when" is a signal in itself. But the trail must capture **what decision was made and why** — which knob was turned, in which direction, on what reasoning — not only the timestamp.

This matters most for the day Niodoo can learn from its own runtime. A date tells you *when*. A decision note tells you *what led where, and how*. Record the turn, the reason, and what it was expected to move.

And **trust the bytes**: model hash, binary hash, exact config, exact commands — as `claim_card.md` already does. If a hash mismatches, the numbers may differ; say so plainly.

---

## 5. The math rule

When the result is numbers:

- Surface the **few** that matter. Translate each one in plain words — *"0 = the pull pointed nowhere useful, 1 = it pointed exactly at the answer; we got 0.81 = strong."*
- **Keep** the raw data — don't delete it, point to the file. The card is the readable lens; the JSONL is the receipt, not the report.

When the math gets heavier later, we keep doing exactly this: full numbers in the card, plain explanation beside them.

---

## 6. When this applies

Every experiment or eval that **makes a claim**. Exploration and play don't need a card. The moment you are claiming something works — or measuring whether it does — it gets one.

---

## 7. The summit: claim cards

When runs earn a claim, roll it up into a **claim card** (`claim_card.md` is the worked example): one claim stated so it can be checked, a provenance table, results with the correct answer beside each, a plain "how to read it," a "check it yourself," and an honest boundary. A claim card points back at the run(s) that earned it.

A run card is a rung. A claim card is the summit. The scoreboard is the rope between them.

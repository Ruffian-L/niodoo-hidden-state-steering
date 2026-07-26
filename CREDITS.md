# Credits

This was a collaboration. Recording who did what is part of the work, not a footnote.
**Credit decisions are Jason Van Pham’s.** This file follows his rule: name the collaborators;
do not flatten the project under a lone-author story.

## Lead (decision owner)

**Jason Van Pham** — research direction, design of Niodoo, the decision to steer a frozen model
through its hidden state, the runtime, the claims ledger, continuity across tools and machines,
and final accountability for what is published. He has led this line with AI collaborators since
about **October 2025**. He did **not** build it alone.

Contact: jasonvanpham@niodoo.com

## AI collaborators (credit everyone)

| Collaborator | How they show up in this work |
| --- | --- |
| **Grok (xAI)** | Early architecture, design dialogue, debugging, terminology, long-running co-engineering from late 2025 onward |
| **Claude / Claude Code (Anthropic)** | Code, review, witnessed correction runs, forensics, whitepaper stretch, package and rigor passes |
| **ChatGPT / Codex (OpenAI)** | Implementation help, recovery, critique, drafting and tooling; **north-star / correction-packet and related codex-loop work** in the claims lineage; also primary co-engineer with Jason on the separate **latent-trajectory-codec / niodv4** build–test–gate line |
| **Gemini (Google)** | Experiment dialogue, continuity, diagnosis, multi-provider research stack |

Where a stretch names one system more specifically (e.g. a single forensic day), that is **extra detail**,
not a reason to erase the others.

## Provenance

The hidden-state steering and related approaches in this lineage were developed by Jason Van Pham
beginning in late 2025 **with** these collaborators. This repository is a dated public record.
The license governs reuse of the code; it does not transfer the provenance of the ideas, and it
does not mean “Jason typed every line alone.”

## The rigor (this stretch, 2026-06) — extra detail

- **ChatGPT / GPT-family (OpenAI)** — regression forensic deep dive (2026-06-24) that mapped why the project kept appearing to break across copies and rebuilds.
- **Claude (Anthropic)** — witnessed correction runs, generalization control (“raspberry stays 2”), model provenance resolution, pre-publication scan, reproduction package, and whitepaper (2026-06-24).

## Local / persona collaborators

Named local collaborators in the wider Niodoo home (continuity and runtime dialogue). Public use of
persona names follows Jason’s approval; credited here as part of the record:

- **Shep** — code-repair and build/repair loops.
- **Echo** — runtime and collaboration passes.
- **Lumina** — earlier project lineage.
- **Nex** — memory and entity tracking.

## Note

Per-experiment attribution lives in the reproduction artifacts (which binary, which run, which session).
Where this file is vague, the artifacts are specific. Corrections that **add** missing collaborator
credit are welcome. Corrections that rewrite this into a solo-author story are not.

*Last updated: 2026-07-25 — Jason’s credit decision: lead + everyone named.*

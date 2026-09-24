# Project Rules & Conventions

## Project: OceanEmbed — Subsurface Ocean Temperature Reconstruction
**SIH Problem Statement ID:** SIH26066 | **Team:** Neural Shadows

Purpose of this document: keep the team aligned on how we work, name things, structure code, and make decisions — so we don't lose hackathon time to avoidable confusion.

---

## 1. Repository Structure

```
oceanembed/
├── data/
│   ├── raw/              # untouched downloaded satellite/reanalysis/ARGO data
│   ├── interim/          # partially processed (regridded, not yet normalized)
│   └── processed/        # final model-ready datasets
├── notebooks/            # exploration only — nothing production-critical lives here
├── src/
│   ├── preprocessing/    # harmonization, regridding, gap-filling
│   ├── embedding/        # encoder models
│   ├── reconstruction/   # decoder / reconstruction models
│   ├── validation/       # metrics, evaluation scripts
│   └── application/      # OHC / anomaly detection logic
├── models/               # saved model checkpoints (not raw weights in git — see §6)
├── demo/
│   ├── backend/          # inference API
│   └── frontend/         # map + profile visualization UI
├── docs/                 # PRD.md, architecture.md, rules.md, references/
└── tests/                # unit tests for pipeline and model components
```

**Rule:** No pipeline logic lives inside notebooks once it's stable — move it into `src/` and import it. Notebooks are for exploration and plotting, not production code.

---

## 2. Data Handling Rules

- **Never commit raw data files** (satellite NetCDFs, ARGO dumps) to git — they're large and not ours to redistribute. Use `.gitignore` on `data/raw/` and `data/interim/`.
- **Document every dataset's source, access date, and version** in `docs/references/data-sources.md` — if we can't reproduce where a file came from, we can't defend it to judges.
- **Never silently interpolate/fill gaps without logging it.** Any interpolation or regridding step must record what was filled and how, so we can honestly answer a judge's question about data quality.
- **Keep a fixed train/validation split** decided early (e.g., specific years or ARGO float subsets held out) — don't let anyone retrain on a different split later without team-wide agreement. Changing splits silently invalidates comparisons across experiments.

---

## 3. Modeling Rules

- **Every architecture choice needs a one-line written justification** in `docs/architecture-decisions.md` — not just "we tried it and it worked better." Judges will ask *why*, and "it scored higher" alone is a weak answer if you can't say why you expected that.
- **Log every experiment** (architecture, hyperparameters, metrics) in a shared sheet or `experiments.md` — no untracked training runs. If we can't reproduce our own best result two days before the demo, that's a self-inflicted problem.
- **Baseline first.** Before any deep learning model, establish a simple baseline (e.g., nearest-neighbor or linear regression from surface to subsurface) so we can honestly quantify how much the deep learning approach actually improves things.
- **Validate against real ARGO data, not just a random train/test split of the same reanalysis product** — the problem statement explicitly calls for independent validation, and it's a stronger judge-facing claim.

---

## 4. Code Conventions

- **Language:** Python throughout (pipeline, modeling, backend).
- **Style:** PEP8, run through a formatter (e.g., `black`) before committing.
- **Naming:** descriptive, not abbreviated-to-the-point-of-confusion — e.g., `reconstruct_temperature_profile()` not `rtp()`.
- **Config over hardcoding:** paths, hyperparameters, and domain bounds (lat/lon box, depth levels) live in a config file, not scattered as magic numbers across scripts.
- **No commented-out dead code left in commits** — delete it; git history keeps it if we need it back.

---

## 5. Git / Collaboration Rules

- **Branch per feature/component** (e.g., `preprocessing-pipeline`, `vit-encoder`, `demo-frontend`) — no direct commits to `main` once the team is past solo-prototyping stage.
- **Pull requests reviewed by at least one other team member** before merging into `main`, even under time pressure — catches integration issues early.
- **Commit messages describe what and why**, not just "fix" or "update".
- **Daily sync** (even 10 minutes) to flag blockers — data pipeline delays are the biggest risk to this project's timeline, so they need to surface immediately, not the night before a deadline.

---

## 6. Model Artifacts

- Large model checkpoints are **not committed to git** — use a shared drive link or `.gitignore` + README pointer instead.
- Every saved checkpoint is named with architecture + date + key metric (e.g., `attn_unet_2026-09-10_rmse0.42.pt`) — no generic `model_final_v2_ACTUALLY_final.pt`.

---

## 7. Demo & Presentation Rules

- The demo must run **offline/locally** as a fallback — never depend solely on live internet access or an external API during the actual judging slot.
- **Every claim shown on screen must be backed by a real computed number** — no placeholder/mock metrics left in by accident. Double-check before the final run-through.
- Rehearse the demo end-to-end at least twice before presenting, including the "what if it breaks" fallback (pre-recorded screen capture as backup).

---

## 8. Honesty & Scoping Rules

- **We do not claim novelty of the core method.** The surface-to-subsurface deep learning approach is established in published literature (see `docs/PRD.md` §7). Our differentiation is regional focus + applied output layer + execution quality — say this plainly if asked, don't oversell.
- **If a component isn't finished by demo day, don't fake it.** Show what's real, explain what's in progress, and describe the plan — judges respond better to honest partial completion than to a discovered fake result.
- **Scope ruthlessly if time runs short.** Priority order if we need to cut: (1) working reconstruction + validation metrics, (2) demo UI with real data, (3) applied OHC/heatwave layer, (4) polish/extra architectures compared. Don't sacrifice #1 or #2 to chase #3 or #4.

---

## 9. Decision Log

Any significant decision (architecture choice, dataset choice, scope cut) gets one line in `docs/decisions.md` with date and reasoning — so nobody re-litigates a settled decision two days before the deadline, and so we can explain our reasoning to judges if asked.

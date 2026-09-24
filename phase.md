# Phased Execution Plan

## Project: OceanEmbed — Subsurface Ocean Temperature Reconstruction
**SIH Problem Statement ID:** SIH26066 | **Team:** Neural Shadows

Purpose: break the build into sequential phases so the team always knows what "done" looks like at each stage, and where the highest-risk work sits (per `rules.md` §8 — scope ruthlessly, protect the core deliverable first).

Note: exact dates aren't filled in — durations shown are relative proportions of the total available time. Fill in real dates once the SIH internal-hackathon / grand finale schedule is confirmed with your mentor.

---

## Phase 0 — Groundwork & Literature Review

**Goal:** Understand the problem deeply before writing pipeline code.

- Read the key prior-art papers (TS-Cast, North Atlantic interpretable DL study, DORS ConvLSTM paper, attention-enhanced 3D U-Net++, ViT-based global framework) — see `docs/PRD.md` §7.
- Identify which surface variables and architectures performed best in comparable regions, and why.
- Confirm exact satellite data sources for each input variable (SST, SSS, SSH/SLA, currents, winds) and their native resolution/format.
- Confirm access to GLORYS reanalysis and INCOIS Gridded ARGO data.
- Finalize domain bounds, depth levels, and resolution (already fixed by problem statement: 5°N–30°N, 45°E–105°E, 0.25°, daily, 15 depths).

**Exit criteria:** Team can explain, in one paragraph each, why we're choosing our initial architecture direction and which datasets we're pulling from — with a source link for each.

---

## Phase 1 — Data Pipeline (Preprocessing & Harmonization)

**Goal:** A clean, aligned, model-ready dataset. This is the highest-risk phase (per `architecture.md` §6) — protect this timeline above all else.

- Ingest raw satellite/reanalysis NetCDF files for all input variables.
- Build regridding pipeline → common 0.25° × 0.25° grid.
- Temporal alignment → daily resolution.
- Crop to North Indian Ocean domain.
- Gap-filling / missing-data handling (document every fill, per `rules.md` §2).
- Normalize/scale variables.
- Join harmonized surface data with GLORYS temperature targets.
- Hold out a fixed ARGO validation subset (by year/season/region — decide and freeze this split, per `rules.md` §2).

**Exit criteria:** A single reproducible script/pipeline that takes raw downloaded data and outputs a model-ready training set + a separate frozen validation set. Runs start-to-finish without manual intervention.

**Checkpoint risk flag:** If this phase overruns significantly, cut geographic scope first (e.g., Bay of Bengal only, not full North Indian Ocean) rather than compromising data quality.

---

## Phase 2 — Baseline Model

**Goal:** Establish an honest reference point before deep learning (per `rules.md` §3).

- Implement a simple baseline (e.g., linear regression or nearest-neighbor mapping from surface variables to subsurface temperature).
- Evaluate baseline against the frozen ARGO validation set.
- Record baseline metrics (correlation, RMSE, bias per depth level) as the number every later model must beat.

**Exit criteria:** A documented baseline score. Every subsequent model comparison references this number.

---

## Phase 3 — Embedding + Reconstruction Model (Core Deliverable)

**Goal:** The actual deep learning system — this is deliverable #1 priority per `rules.md` §8.

- Implement first candidate architecture (start with the simplest defensible option — likely CNN or attention-enhanced U-Net per prior art).
- Train on harmonized dataset from Phase 1.
- Evaluate against frozen ARGO validation set; compare to Phase 2 baseline.
- Log every experiment (architecture, hyperparameters, metrics) per `rules.md` §3.
- If time allows, try a second architecture (e.g., ViT or attention-hybrid) for comparison — this is a stretch goal, not core scope.

**Exit criteria:** A trained model that measurably outperforms the baseline, with logged metrics broken down by depth level, and a one-line written justification for the chosen architecture (`docs/architecture-decisions.md`).

---

## Phase 4 — Validation & Error Analysis

**Goal:** Turn raw metrics into an honest, presentable evaluation story.

- Full validation run against independent ARGO profiles.
- Depth-wise, seasonal, and regional error breakdown.
- Identify and document where the model performs well vs. poorly — this honesty is a judge-appeal point, not a weakness to hide (per `rules.md` §8).

**Exit criteria:** A clear evaluation summary (charts + numbers) ready to present, including known limitations.

---

## Phase 5 — Application Layer (Ocean Heat Content / Anomaly Detection)

**Goal:** Convert raw reconstruction into a decision-relevant signal — this is what separates OceanEmbed from a pure research replication.

- Implement Ocean Heat Content (OHC) estimate from reconstructed temperature profiles.
- Implement marine heatwave / thermal anomaly flagging logic (e.g., deviation from seasonal climatology).
- Validate that flagged anomalies correspond to known/plausible events where possible.

**Exit criteria:** A working function that takes a reconstructed field and outputs an OHC value and/or anomaly flag per location/day.

**Priority note:** Per `rules.md` §8 scope order, this is priority #3 — if time runs short, a simplified version (OHC only, no anomaly detection) is an acceptable cut.

---

## Phase 6 — Demo UI & Integration

**Goal:** A live, visual, working demo — priority #2 per `rules.md` §8.

- Build map interface (Bay of Bengal / Arabian Sea).
- Click-to-query: user selects a location → live-generated temperature-vs-depth profile.
- Side-by-side comparison view: reconstructed profile vs. real ARGO reading (for validation locations).
- Anomaly/heatwave overlay on the map.
- Ensure demo runs fully offline/locally (per `rules.md` §7).

**Exit criteria:** End-to-end demo runs without needing live internet access, using real precomputed data (no placeholder/mock numbers, per `rules.md` §7).

---

## Phase 7 — Rehearsal & Presentation Prep

**Goal:** De-risk the actual judging moment.

- Full dry-run of the demo, timed to the actual pitch slot length.
- Prepare fallback (pre-recorded screen capture) in case of live technical failure.
- Prepare answers to likely judge questions:
  - "Why this architecture?"
  - "How is this different from existing research?" (answer honestly per `rules.md` §8 — regional focus + applied layer + execution)
  - "What are the model's limitations?" (reference Phase 4 error analysis)
  - "Could this actually be deployed by INCOIS?"
- Second rehearsal after incorporating feedback from the first.

**Exit criteria:** Team can deliver the full pitch confidently within the time limit, handle at least the four questions above, and has a working fallback if live demo fails.

---

## Priority Summary (if time runs short)

Per `rules.md` §8, in order of what must survive:
1. Working reconstruction model + honest validation metrics (Phases 1–4)
2. Working demo UI with real data (Phase 6)
3. Applied OHC/heatwave layer (Phase 5)
4. Multiple architecture comparisons / extra polish (stretch goals within Phase 3)

---

## Open Items to Confirm with Mentor

- Actual SIH internal hackathon and grand finale dates, to convert phase proportions into real deadlines.
- Whether a mid-point check-in / progress review is required, and when.
- Compute resource availability, which affects how ambitious Phase 3's architecture experiments can be.

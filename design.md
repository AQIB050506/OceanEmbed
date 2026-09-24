# Design Document

## Project: OceanEmbed — Subsurface Ocean Temperature Reconstruction
**SIH Problem Statement ID:** SIH26066 | **Team:** Neural Shadows

Purpose: define the look, feel, and interaction design of the demo application — the part judges actually see and touch. Technical pipeline design lives in `architecture.md`; this document covers the presentation layer.

---

## 1. Design Goals

- Make an abstract concept (3D subsurface temperature reconstruction) **immediately understandable** within seconds of a judge looking at the screen.
- Prioritize the **"wow moment"**: input → AI processing → visible, credible outcome (per SIH judging criteria on demo potential).
- Be honest in visualization — never imply more precision or certainty than the model actually has (ties to `rules.md` §8 honesty principle).
- Work fully offline with real precomputed data (per `rules.md` §7) — no dependency on live internet during judging.

---

## 2. Primary User (Judge) Journey

1. Judge sees a **map of the Bay of Bengal / Arabian Sea** with a subtle overlay (e.g., current sea surface temperature).
2. Judge (or presenter) **clicks a location** on the map.
3. A **temperature-vs-depth profile chart** appears — showing the model's reconstructed curve.
4. If that location has a real ARGO validation point, the **actual measured profile is overlaid** — visually proving the reconstruction is close to reality.
5. Presenter switches to an **anomaly/heatwave overlay view** — showing flagged regions of unusual subsurface warming across the whole basin.
6. Presenter narrates the real-world hook: *"This is the kind of signal that feeds into cyclone intensification forecasts and marine heatwave alerts."*

This journey should take under 90 seconds end-to-end — it needs to work within a short demo slot.

---

## 3. Screen-by-Screen Design

### 3.1 Landing / Map View (Primary Screen)
- **Base layer:** Map of North Indian Ocean domain (5°N–30°N, 45°E–105°E), Bay of Bengal and Arabian Sea both visible.
- **Default overlay:** Current reconstructed sea surface / near-surface temperature, as a smooth color gradient (cool blue → warm red/orange).
- **Interaction:** Clickable/tappable — any point on the ocean surface is queryable.
- **Toggle control:** Switch between "Temperature Overlay" and "Anomaly/Heatwave Overlay" views.
- **Minimal chrome:** no unnecessary UI clutter — the map and data should dominate the screen.

### 3.2 Depth Profile View (triggered on click)
- **Layout:** Side panel or modal, not a full page navigation — keep the map visible for context.
- **Chart type:** Line chart, X-axis = temperature (°C), Y-axis = depth (m), **inverted** (0m at top, 1000m at bottom — matches physical intuition of "going deeper").
- **Two lines when available:**
  - Solid line: model-reconstructed profile.
  - Dashed/marker line: real ARGO measurement (only shown at validation locations — must be labeled clearly as "Actual ARGO Reading" so it's never confused with another prediction).
- **Annotation:** Show correlation/RMSE for that specific point if it's a validation location — small, unobtrusive text, not a dominant number.
- **Location label:** Latitude/longitude and date of the shown profile, clearly visible.

### 3.3 Anomaly / Heatwave Overlay View
- **Visualization:** Map recolored to highlight regions where subsurface temperature deviates significantly from seasonal norms (e.g., warm anomaly regions in red/orange, normal in neutral tone).
- **Purpose:** This is the "so what" screen — the one that connects the technical work to cyclone/fisheries/ecosystem relevance without needing the judge to read numbers.
- **Optional label:** A short caption like "Elevated subsurface heat content — relevant to cyclone intensification risk" tied to flagged regions, if time permits building this logic (Phase 5, `phase.md`).

### 3.4 "How It Works" Supporting Screen (secondary, for Q&A)
- A simple, non-technical diagram (input satellite variables → embedding → reconstruction → validation) for when judges ask "how does this actually work?"
- Should mirror the architecture diagram in `architecture.md` but simplified — no code-level detail, just the conceptual flow.
- Include the honest note about prior art / related research (per `rules.md` §8) if a judge asks about novelty — better to have this ready than to look caught off guard.

---

## 4. Visual Design Direction

- **Tone:** Scientific, credible, calm — not flashy sci-fi. This is a real earth-science tool, not a game. Overly stylized visuals can undercut credibility with domain-aware judges.
- **Color palette:**
  - Ocean/temperature gradient: standard oceanographic convention — blue (cold) → red/orange (warm). Don't invent a novel color scheme; using the convention scientists expect signals domain awareness.
  - Neutral background (dark navy or muted grey) so the temperature data itself is the visual focus.
  - Anomaly overlay: a distinct, high-contrast color (e.g., bright red/amber) so flagged regions are unmistakable at a glance.
- **Typography:** Clean, legible sans-serif. Avoid decorative fonts — this is a scientific tool, legibility matters more than personality.
- **Motion:** Minimal — a smooth transition when switching overlays or opening the profile panel is fine; avoid gratuitous animation that adds no information.

---

## 5. Accessibility & Clarity Considerations

- Ensure color choices are colorblind-safe where possible (avoid pure red/green as the only distinguishing signal) — use a perceptually uniform colormap (e.g., viridis-style) as a fallback option if standard warm/cold gradient risks ambiguity.
- All charts must have clear axis labels and units — never assume a judge remembers what "0.25°" or "depth level" means without a label reminding them.
- Keep text minimal on the map itself; put detail in the side panel, not cluttering the visualization.

---

## 6. Technical Notes for Implementation (cross-reference `architecture.md`)

- Frontend should query **precomputed** reconstruction outputs, not run live inference during the demo, to avoid latency or failure risk on stage. Live inference can be a background "nice to have" if performance allows, but the safe path is precomputed results for the specific demo locations/dates chosen in advance.
- Map library: Leaflet or Mapbox (per `architecture.md` §5 tech stack).
- Chart library: any standard JS charting library capable of line charts with dual series (e.g., Chart.js, Plotly).
- Keep the demo dataset (precomputed profiles + validation points) bundled locally so the whole app runs without internet (per `rules.md` §7).

---

## 7. What NOT to Do

- Don't show raw correlation/RMSE numbers as the headline of the demo — they mean little to a general judge audience. Lead with the visual profile match, mention numbers as supporting evidence.
- Don't over-promise precision — avoid phrasing or visuals that imply the model is "measuring" the ocean; it's *reconstructing/estimating* it. This distinction matters for scientific credibility.
- Don't clutter the first screen judges see — the map + temperature overlay should be simple and immediately readable before any interaction happens.
- Don't hide the ARGO-vs-reconstructed distinction — conflating "real" and "predicted" data, even visually, undermines trust the moment a sharp judge asks "wait, is that real or predicted?"

---

## 8. Open Design Questions (to confirm with mentor/team)

- Should the demo include a date/time slider to show reconstruction changing over time, or stick to fixed pre-selected dates for reliability?
- How many validation (ARGO-comparison) locations should be pre-selected and bundled for the demo — enough to look robust, but curated enough to guarantee good-looking results within available time?
- Should the anomaly/heatwave overlay be a core Phase 6 deliverable or treated as a stretch feature if Phase 5 application-layer work runs behind schedule?

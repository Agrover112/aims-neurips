# Evidence manifest

## Manuscript

- manuscript path: `docs/raa_irt_design.md`
- manuscript version: method-design note, 2026-10-17
- target venue: AIMS predictive AI evaluation competition at NeurIPS 2026
- venue brief: `paper/neurips_2026/venue_brief.md`

## Audit Command

- artifact root: `/home/agrov/aims-neurips`
- command: `python3 /home/agrov/.codex/skills/research-paper-writer/scripts/audit_evidence_manifest.py --manifest paper/neurips_2026/evidence_manifest.md --root /home/agrov/aims-neurips`
- audit result: rerun recorded during the design-note review
- audit report path: terminal output only

## Runs

- result id: none
- command: not applicable
- data version: not applicable
- code revision: not applicable
- config: not applicable
- checkpoint/model version: not applicable
- seed: not applicable
- environment: not applicable
- output artifacts: `docs/raa_irt_design.md` records that no model runs exist

## Figures

- figure id: model-family-schematic
- raw data: `docs/raa_irt_design.md`; conceptual architecture
- generation script/command: include `paper/neurips_2026/figures/irt_model_family.tex` in LaTeX
- output file: `paper/neurips_2026/figures/irt_model_family.tex`
- sample set: not applicable
- key claim: distinguishes the dependencies in IRT, AIRT, published GPIRT,
  CNP-AIRT, NP-AIRT, ANP-AIRT, and proposed RAA-IRT
- verification status: checked
- verification scope: source reviewed; TeX compilation pending because no TeX
  engine is installed

- figure id: raa-irt-schematic
- raw data: `docs/raa_irt_design.md`; conceptual architecture
- generation script/command: include `paper/neurips_2026/figures/raa_irt_model.tex` in LaTeX
- output file: `paper/neurips_2026/figures/raa_irt_model.tex`
- sample set: not applicable
- key claim: shows fixed item parameters, global conditioning, and bounded
  target-specific relational correction
- verification status: checked
- verification scope: source reviewed; TeX compilation pending because no TeX
  engine is installed

## Tables

- table id: matched-model-ladder
- raw data: `docs/raa_irt_design.md`
- calculation script/command: `docs/raa_irt_design.md`; no calculation required
- output file: `docs/raa_irt_design.md`
- reported metrics: none
- key claim: identifies the component isolated by each proposed comparison
- verification status: checked

## Formulas

- equation id: amortized-2pl-base
- exact implementation or external source: `docs/raa_irt_design.md`; conceptual
  2PL specialization of AmortizedIRT
- linked run/config: none
- symbol definitions: `docs/raa_irt_design.md`
- verification status: checked

- equation id: map-ability-update
- exact implementation or external source: `docs/raa_irt_design.md`; regularized
  2PL log-posterior
- linked run/config: none
- symbol definitions: `docs/raa_irt_design.md`
- verification status: checked

- equation id: relation-aware-adapter
- exact implementation or external source: `docs/raa_irt_design.md`; proposed
  equation
- linked run/config: none
- symbol definitions: `docs/raa_irt_design.md`
- verification status: checked

## Claims

- claim id: np-test-time-conditioning
- manuscript location: `docs/raa_irt_design.md`, “Does an NP perform test-time
  adaptation?”
- supporting figures/tables/artifacts/logs: `docs/raa_irt_design.md` and
  `reference/starting_kit/README.md`
- evidence strength: established mechanism, not empirical competition evidence
- verification status: verified

- claim id: raa-improvement
- manuscript location: none; deliberately not claimed
- supporting figures/tables/artifacts/logs: `paper/neurips_2026/evidence_manifest.md`
- evidence strength: unsupported
- verification status: verified

## References

- citation key: garnelo2018np
- verified bibliographic source: https://arxiv.org/abs/1807.01622
- sentence or claim supported: NP uses context-conditioned latent inference
- source support note: primary paper
- verification status: verified

- citation key: kim2019anp
- verified bibliographic source: https://arxiv.org/abs/1901.05761
- sentence or claim supported: ANP adds target-specific attention
- source support note: primary paper
- verification status: verified

- citation key: duckmayr2020gpirt
- verified bibliographic source: https://proceedings.mlr.press/v124/duck-mayr20a.html
- sentence or claim supported: published GPIRT places GP priors on item response
  functions indexed by ability
- source support note: primary paper and proceedings page
- verification status: verified

- citation key: huang2023rcnp
- verified bibliographic source: https://arxiv.org/abs/2306.10915
- sentence or claim supported: Relational Conditional Neural Processes is an
  existing method name
- source support note: primary paper
- verification status: verified

- citation key: dubois2020npf
- verified bibliographic source: https://github.com/YannDubs/Neural-Process-Family
- sentence or claim supported: repository implements CNP, latent NP, attentive
  variants, and convolutional variants
- source support note: implementation repository and MIT license
- verification status: verified

## Open Gaps

- unsupported items: any claim that RAA-IRT improves Brier score, calibration,
  uncertainty, or acquisition
- conflicts between artifacts: the README's old numerical table is explicitly
  fabricated and cannot be used as evidence
- pending experiments: all AIRT/CNP/NP/ANP/GP-adapter/RAA matched comparisons
- claims that must be weakened, removed, or deferred: performance, novelty,
  calibration, and runtime claims

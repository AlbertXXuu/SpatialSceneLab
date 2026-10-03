# Object-level editing: next-stage evaluation protocol

Revision: 2026-10-04.1. Status: **S0 development comparison executed; full protocol pending**.

The [S0 report](../reports/DEVELOPMENT-S0-2026-10-04.md) records implemented
direct controls and current nearest-reference matching. Visibility-aware matching,
held-out/real evidence and full editing gates remain pending. Targets are unchanged.

[中文](NEXT-STAGE-PROTOCOL.zh-CN.md) · [Measured baseline](../reports/LOCAL-SCAN-2026-10-04.md)

## Scope and observed problem

The next deliverable is a local Blender workflow: prepared RGB-D capture with
known calibration/poses and object-box priors → reconstructed observed surfaces
→ object partition and uncertain boundary review → user correction → translation
and rotation → save, reopen and export an editable scene.

The baseline retains 275 candidate Sofa0 boundary triangles in the environment.
These are **candidates, not 275 verified labeling errors**. The current public
fixture has two simple boxes; the real sample is a single scan with no independent
instance ground truth. The next evaluation addresses object-selection quality,
editing consequences and task completion. Camera recovery from arbitrary RGB,
unseen-surface recovery and physical simulation are separate capabilities.

## Research question and hypothesis

With the same reconstructed mesh, can weak box priors, surface adjacency and
multi-view observation evidence improve selection of a target's observed surface
without moving more non-target surface? Can explicit uncertainty and limited
user correction reduce cleanup effort?

The proposed evidence-guided partition is a hypothesis. Start with the direct
baselines below. Add graph optimization or another dependency only when their
recorded failure establishes its purpose. Keep uncertain ownership explicit.
Evaluate automatic output and assisted output separately; an assisted result is
not evidence of automatic segmentation performance.

## Scene matrix and acquisition

The planned main experiment has **24 independent authored scene groups**:
6 development groups and 18 held-out test groups. This is a construction target,
not a statistical sufficiency claim. First implement six varied development
groups. Use pilot variance to plan statistical precision; if more test groups
are needed, decide that before inspecting held-out method comparisons.

| Factor | Cases to construct | Evidence recorded |
| --- | --- | --- |
| Geometry | curved/concave objects, chair legs, shelves, small objects | instance geometry, surface area and minimum feature width |
| Contact and proximity | floor/wall contact, adjacent furniture, overlapping boxes | contacts, separation in meters and box overlap |
| Appearance | similar colors, color discontinuities within one object | material assignments and controlled appearance changes |
| Visibility | partial occlusion, narrow coverage, sparse trajectories | frame count, visible target area and unobserved fraction |
| Sensor and priors | depth noise/dropout, pose error, incorrect box extent/orientation | noise in meters, angles and missing-data ratios |
| Combined stress | contact + same color + occlusion; thin geometry + noisy depth | individual factor levels and their combination |
| Task sequences | translation, rotation, two-object edits, correction followed by undo/reopen | task specification, pivot, transforms and expected state |

Single-factor experiments isolate causes; combined cases test interactions. Use
multiple furniture geometries and room layouts, not recolored copies of two boxes.
Perturbation ranges must come from documented input behavior, development
measurements or a stated stress assumption, then be fixed before test execution.

The external-validation acquisition target is **six independent real rooms**,
including at least four with usable instance-reference annotations. Admit a room
only after checking its access terms, format and reference eligibility. This
target is pending acquisition. The existing unlicensed tea_room remains a local
qualitative stress case and does not satisfy the public-data or independent-label
gate. Missing real inputs leave real-scene generalization acceptance pending;
the authored-data experiment can still be reported under its own scope.

Original authored inputs/outputs should have explicit redistribution terms.
External data and derived assets retain their own terms. ScanNet is a possible
annotated reference source, but its official repository requires a data-access
agreement; it is not a ready-to-bundle dependency. No access request, agreement
or redistribution is implied by this protocol.

## Independence and reference construction

Split by complete room/asset/trajectory group. Exact meshes, layouts, repeat scans,
frames, noise seeds and perturbations from a group stay in one split. Do not count
variants as additional independent scenes. Development labels select parameters;
held-out labels are evaluation-only. A new version that tunes on a test result
needs a fresh held-out comparison before making a generalization claim.

All methods receive the same RGB-D, poses, boxes, input-derived candidate region
and TSDF mesh. Physical geometry and ray labels are stored in an evaluation-only
reference bundle. Map area-uniform mesh samples to independently labeled visible
surfaces using a documented distance/visibility rule. Freeze matching tolerances
on development data and publish sensitivity to them. Samples with no defensible
reference stay unscorable; report their area and conservative bounds. Never use
the tested box assignment to create its own ground truth.

Real annotations must state their origin, uncertain regions and review process.
Record label reliability as unverified if no independent review is available.
Distinguish correctness on reconstructed observed surfaces, reference-visible
coverage and complete-object coverage. No score implies an unseen backside has
been reconstructed.

## Comparisons

| ID | Baseline | Purpose |
| --- | --- | --- |
| B0 | current three-vertices-agree ownership | preserve the released reference |
| B1 | unique-box assignment at triangle centroid | test whether a simple decision rule suffices |
| B2 | within-face area sampling or box-boundary clipping | separate discretization artifacts from inference errors |
| B3 | seeded geometry-based region growth | compare adjacency/normal continuity without multi-view evidence |
| B4 | cue-based interactive graph partition | compare a stronger classical approach with equal correction budget |
| Oracle | independent reference ownership on the same mesh | estimate the partition ceiling; never presented as a usable method |

Freeze preprocessing, region size, seed budget and allowed correction actions.
Publish B4's exact energy/implementation; a mesh adaptation is not a reproduction
of an RGB-D image paper. Survey the closest published 3D/interactive method for
a research contribution claim. Run it when its data/resource contract is feasible;
otherwise explain the missing comparison and limit the claim rather than asserting
state-of-the-art performance. Resource feasibility cannot establish superiority.

## Metrics and statistical unit

Use surface area, not point/triangle counts, as the primary weighting. Partition
IoU/precision/recall and edit-area ratios use only independently reference-labeled,
scorable area on the **same fixed reconstructed mesh**. Their target denominator
is that mesh's scorable target area, including target area outside the input-derived
candidate region; those missed areas still count as false negatives. Report
reference-visible reconstruction coverage on the independent reference surface
separately. Do not mix physical-reference area with partition denominators.
Unscorable area and its conservative bounds are reported separately; the
risk/coverage candidate-region denominator is distinct from the full-mesh domain.

| Metric | Definition / reporting |
| --- | --- |
| Target IoU, precision, recall | area-weighted against independent target labels; unresolved target area counts as missed selection |
| Missed edit | scorable target mesh area not assigned the specified transform / full scorable target mesh area |
| Wrong edit | moved scorable non-target mesh area / full scorable target mesh area; also break out walls, floor and other objects |
| Risk and coverage | incorrect accepted ownership area / accepted ownership area, alongside accepted area / fixed scorable candidate-region area; report full curves and reference-label availability |
| Geometric edit fidelity | area-uniform samples versus ideal rigid transform, p50/p95/max distances in meters; unrelated geometry drift |
| Assisted task cost | correction strokes/actions, elapsed time, undo/reopen success and final quality under the same budget |
| Runtime and resources | process-tree peak RAM, relevant VRAM, wall time and interactive response by workload size |
| Export fidelity | world geometry, topology, UUID, units, ownership and color/material checks after native reopen |

The predeclared primary automatic endpoint is scene-macro target IoU. Wrong-edit
area is a safety endpoint; freeze its non-inferiority margin from development
task tolerance before test execution. Report risk/coverage and per-factor results
even when the scalar endpoint is favorable. A claim that the proposed method
improves the baseline requires a positive lower bound on the paired scene-level
95% confidence interval against the development-selected strongest direct
baseline, and passage of the predeclared wrong-edit guard. This is this project's
claim gate, not a universal publication rule.

Use paired scene-level bootstrap intervals and publish scene-wise differences.
Multiple noise seeds remain repeated measurements within a scene. Surface samples,
triangles and frames are not independent experimental subjects. Include crashes,
OOM, empty output and timeouts in attempted-task denominators, with missing scores
and task failures reported explicitly. Do not silently drop them from averages.
If precision is insufficient or the interval includes no improvement, report an
inconclusive/negative result rather than a success claim.

## Ablations and failure cases

Ablate observation visibility/provenance, adjacency, normal/color cues, uncertainty
and boundary clipping one at a time, retaining the same tuning allowance. Plot
quality against correction effort and report automatic and assisted results
separately. Simulated correction uses a fixed documented seed/action policy;
timed author operation is labeled as author testing, not a user study.

Include same-color touching objects, coplanar object/wall surfaces, almost invisible
targets, inaccurate/missing boxes, occluded thin structures, zero valid depth,
invalid poses and output-write failure. Record best, typical and worst cases.
Uncertainty must remain visible in saved state and exports. Collision/open-boundary
warnings describe observed surfaces; they do not certify watertight solid occupancy.

## Freeze, reproduction and claims

Before held-out evaluation, commit the scene manifest/split, reference construction,
metrics, tolerances, methods, ablations, interaction budgets and failure policy.
Use existing requirements and native Blender controls where sufficient. Run
prepared inputs locally with zero new paid services; record actual OS-level
network isolation separately from the existing Python network guard.

Deliver per-scene machine results, exact configurations, raw failure records,
scripts rebuilding tables/plots, and English/Chinese reports. Clearly separate
method evaluation, implementation reproducibility and task usability. Paper-quality
evidence does not by itself establish novelty or publication acceptance. A useful
engineering workflow remains assessable even when the research hypothesis fails.

Official references checked 2026-10-04:

- [NeurIPS paper checklist](https://neurips.cc/public/guides/PaperChecklist): claim scope, experimental detail, reproducibility and statistical reporting.
- [ScanNet official repository](https://github.com/ScanNet/ScanNet): annotated RGB-D inputs and data access/terms.
- [Interactive Segmentation on RGBD Images via Cue Selection, CVPR 2016](https://openaccess.thecvf.com/content_cvpr_2016/html/Feng_Interactive_Segmentation_on_CVPR_2016_paper.html): related classical cue-based graph-cut work; not a novelty claim for this project.

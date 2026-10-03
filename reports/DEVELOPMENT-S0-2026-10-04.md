# Six complex scenes: fixed-mesh partition development results

Measured 2026-10-04 · S0 development evidence

[中文](DEVELOPMENT-S0-2026-10-04.zh-CN.md) · [Reproduction](../docs/DEVELOPMENT-S0.md) ·
[JSON](development-s0-results.json) · [CSV](development-s0-metrics.csv)

## Result and decision

Six independently authored development layouts, 92 RGB-D views and 18 B0/B1/B2
tasks completed on CPU, producing 386,001 sampled observations and 491,066
reconstructed triangles. The simple centroid rule B1 achieved higher target
area IoU than released B0 in all six cases; B2 area sampling did not exceed B1.
Retain B1 as the strong simple control and investigate its remaining errors
before adding more complex inference. This is development evidence, not held-out
superiority or a new segmentation algorithm claim.

| Method | Scene-macro target IoU | Missed target area / target area | Non-target selected area / target area | Evaluable targets |
| --- | ---: | ---: | ---: | ---: |
| B0: three vertices agree | 0.791204 | 20.8548% | 0.029689% | 6 / 6 |
| B1: unique centroid ownership | 0.823510 | 17.5341% | 0.134835% | 6 / 6 |
| B2: within-face area majority | 0.818533 | 18.0361% | 0.129465% | 6 / 6 |

B1 improves this development macro IoU by 3.2306 percentage points, with
increased non-target selection. No non-inferiority margin or held-out statistical
conclusion has been established. Review each scene's errors and unscorable area.

![B0/B1/B2 area comparisons](development-s0/baseline-comparison.png)

## Inputs and independent reference

![Authored physical references; blue identifies each target](development-s0/development-scene-overview.png)

The overview depicts authored reference geometry, not reconstructed completeness.
These original MIT assets are distinct from the unlicensed real tea_room kept
local in the earlier report. Geometry includes curved multi-part furniture,
legs/slats/shelves, contacts, box overlap, similar colors and occlusion.

| Development group | Views | Mesh triangles | Target IoU B0 / B1 / B2 | Scorable mesh area |
| --- | ---: | ---: | --- | ---: |
| Contact chair | 16 | 99,511 | 0.780015 / 0.826313 / 0.810374 | 99.9651% |
| Same-color dining | 18 | 96,518 | 0.632621 / 0.659131 / 0.656372 | 99.9469% |
| Curved sofa / wall | 16 | 67,055 | 0.978251 / 0.989665 / 0.988877 | 99.9520% |
| Occluded bookshelf | 12 | 52,131 | 0.663292 / 0.724015 / 0.721101 | 99.8090% |
| Thin rack / noise | 16 | 103,884 | 0.881332 / 0.914942 / 0.908714 | 99.9598% |
| Combined study | 14 | 71,967 | 0.811715 / 0.826998 / 0.825758 | 99.8314% |

The frozen configuration records geometry, seeds, box perturbations and sensor
stress. Levels are declared development assumptions, not real-sensor calibration.
The generator independently ray casts physical geometry, adds depth noise/dropout,
and writes measured camera poses separately from the true ray-casting poses.
Reference instance labels never come from the tested boxes. All three partitions
are computed before reference labels are loaded.

## Fixed mesh and scoring

Each scene shares a TSDF mesh (0.035 m voxel, 0.105 m truncation) and ownership
priors. B0 requires three vertices to agree; B1 uses the triangle centroid; B2
uses 49 deterministic area samples and accepts a label only if its fraction is
at least 0.6. Unresolved predictions on reference-target area count as missed.

Evaluation uses seven area quadrature samples per reconstructed triangle and a
nearest physical-reference surface within 0.08 m. Competing owner labels whose
nearest distances differ by at most 0.002 m are unscorable. Area IoU is
TP/(TP+FP+FN); both error ratios use scorable target area as denominator. Target
area outside a candidate domain still counts as missed. These measure selection,
not an executed edit or complete-object recovery.

The scorable fractions cover the whole reconstructed mesh, much of it environment.
They do not measure visible-reference coverage or unseen surfaces. The matcher has
no normal/visibility filtering, and physical references contain potentially hidden
part surfaces. Bounds address unscorable labels, not finite quadrature error.
Real-scene physical accuracy remains untested.

## Resources, integrity and failure

The [sensitivity replay](development-s0/sensitivity-results.json) completed
108 / 108 evaluations with no integrity or evaluation findings. All 18 original
0.08 m / 7-sample method tasks matched again, checking every object's metric.
Macro IoU ordering remained B1 > B2 > B0 across these six configurations:

| Distance tolerance | Area samples / face | B0 | B1 | B2 |
| --- | ---: | ---: | ---: | ---: |
| 0.04 m | 7 | 0.793307 | 0.826493 | 0.821256 |
| 0.08 m | 7 | 0.791204 | 0.823510 | 0.818533 |
| 0.12 m | 7 | 0.791120 | 0.823475 | 0.818498 |
| 0.04 m | 49 | 0.793428 | 0.826495 | 0.821272 |
| 0.08 m | 49 | 0.791219 | 0.823470 | 0.818495 |
| 0.12 m | 49 | 0.791135 | 0.823432 | 0.818457 |

The largest target IoU change from 7 to 49 samples at fixed scene/method/tolerance
was 0.000826593 (thin-rack B0, 0.04 m). Scorable full-mesh area ranged
96.5906–99.9691%. This is observed quadrature stability on development inputs,
not an error bound or reference-visible coverage. The replay took 303.859 s
including its CPython guard wrapper; its full configuration results and errors
remain in the linked JSON/CSV.

The final run took 30.760 s including reconstruction and evaluation. Single-process
lifetime peak working set was 240,291,840 bytes (229.16 MiB), not per-case
incremental or full process-tree memory. B0 partition took 0.016–0.032 s, B1
0.031–0.052 s, B2 0.810–1.525 s. This is one local run, not a latency distribution.
The Windows host has 64 GB RAM / RTX 4060 Laptop 8 GB; execution used CPU, no GPU
model inference and zero new paid allocation. Exact versions and hashes are in JSON.

SHA-256 anchors link generator/config, raw inputs, scenes, surface manifests/meshes,
references, face labels and metrics. CPython socket/DNS denial passed its probe;
no denied runtime network event occurred. This is not OS isolation. Earlier two
successful runs remain local as pre-final provenance attempts; published metrics
use the final run with complete replay anchors.

An extra combined-study Blender probe moved the B0 desk region by (0.28, 0, 0.12) m.
Native `.blend` reopen passed. Both GLB reimports failed the environment connectivity
check despite unchanged triangle counts and maximum world-vertex distance 1.284e-6 m.
The [failure JSON](development-s0/native-export-failure.json) is retained. Diagnose
exporter connectivity versus ambiguous verifier nearest-vertex mapping before
accepting this export. A [read-only before-asset diagnosis](development-s0/before-diagnostic.json)
localized its failure to unstable nearest-vertex mapping: two source vertices
only 1.609e-6 m apart caused all five face-index differences, while the local
triangle-corner coordinate multisets were exactly equal. The after asset and
seam/welding semantics remain unverified; original failed results are unchanged.
It is not a passed complex editing demo. Reproduce with
`blender_scene.py`, the combined-study surface, target `combined-desk`, translation
`0.28 0 0.12`.

## Acceptance and next step

S0 delivers complex development inputs, independent labels, fixed-mesh controls,
area metrics and explicit ambiguity/failure accounting. The 18 held-out groups,
real-room instance references, visibility-aware matching, B3/B4 controls, full
editing matrix, correction cost and whole-machine disconnected test remain
pending under the [full protocol](../docs/NEXT-STAGE-PROTOCOL.md).

Next diagnose the GLB failure, then build Blender partition review/correction,
translation/rotation, undo and reopen. Investigate same-color dining and bookshelf
omissions with independent labels, then freeze the final method before inspecting
held-out labels. Paper evidence, method improvement and product tasks have separate
acceptance gates.

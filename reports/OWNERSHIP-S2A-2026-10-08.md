# S2a: explain ownership omissions, then edit the measured B1 partition

Measured 2026-10-08 (Asia/Shanghai). [中文](OWNERSHIP-S2A-2026-10-08.zh-CN.md).

The same missed-area metric hides different assignment mechanisms. On the six
frozen S0 development meshes, **99.46% of the dining target's missed, scorable
area lies in competing target boxes**, while **99.19% of the shelf target's
missed area falls outside its target box and is assigned to the environment**.
These are decision-path diagnoses of the existing B1 rule. Its predictions,
reference matcher and scores have not changed.

The evaluated B1 partition now enters the native Blender workflow. All six
exports, scripted editing tasks and independent native source-lineage checks
passed. A [downloadable dining task](../examples/ownership-b1/dining-b1-author-task.zip)
opens directly in Blender; actual author-operated correction remains pending.

## Fixed inputs and method

The anchor is the published [S0 result](development-s0-results.json), SHA256
`5bb302eac773d8a2e0af299e6fe3f476f413740e4ffd5261419332621c7e2071`.
We reuse six meshes (491,066 triangles), input box/structure priors and recorded
B1 owner arrays. There is no reconstruction rerun, model inference or tuning.
The prior-only replay must exactly recover the original owners before reference
labels enter the diagnostic. B1 export never reads those reference labels.

The unchanged evaluator uses seven barycentric samples per face, a 0.08 m
nearest-reference tolerance and 0.002 m ambiguity margin. Temporary batch-local
face IDs recover per-source-face reference area in bounded batches of 1,024.
Eleven mutually exclusive area categories cover TP, five FN mechanisms, two FP
mechanisms, TN and two unscorable categories. A face may span multiple sampled
reference labels, so areas are retained instead of assigning a majority label.
Structure exclusion takes precedence over overlapping target boxes; outside-box
cases are distinguished by their resulting owner. The raw flags are also saved.

## Observed result

Percentages below divide by that target's **missed, scorable surface area**, not
by all target area or the complete physical object. FN area uses square metres.

| Development target | Unchanged B1 IoU | FN area (m²) | Target-box overlap | Outside target box → environment | Other FN causes |
|---|---:|---:|---:|---:|---:|
| Contact chair | 0.826313 | 0.192059 | 96.1728% | 0% | 3.8272% |
| Same-colour dining | 0.659131 | 0.844693 | 99.4648% | 0% | 0.5352% |
| Curved sofa / wall | 0.989665 | 0.034715 | 0% | 99.7265% | 0.2735% |
| Occluded shelf | 0.724015 | 1.206119 | 0% | 99.1850% | 0.8150% |
| Thin rack / noise | 0.914942 | 0.062575 | 0% | 100% | 0% |
| Combined study | 0.826998 | 0.482964 | 13.7913% | 84.0313% | 2.1773% |

![Missed-area decision paths on all six development targets](ownership-s2a/ownership-failure-causes.png)

All original confusion areas, unscorable areas and TP/FP/FN totals reconcile.
The maximum absolute difference is **6.8213e-13 m²**. Diagnostic execution took
**16.3078 s**, with **228,728,832 bytes** process-lifetime peak RAM. These are
single-run resource observations, not speed comparisons. New paid allocation
and model calls were both zero.

This distinguishes two concrete next experiments: resolve genuinely competing
ownership in the dining scene, and recover observed target surface beyond an
incomplete box in the shelf scene. It does not establish that region growing,
colour cues or a learned method will improve either case.

## Editing and lineage acceptance

The exporter keeps every original face once, with exact serialized PLY geometry,
normals and observed colours. Overlap remains a separate **Unresolved (-2)**
region. Per-part face/vertex maps bind it to the original surface, including
unresolved counts of 1,216 / 3,605 / 0 / 67 / 0 / 877 in the table's scene order.

The unchanged S1 runner completed **6/6 tasks, 24 committed operations, 24
undo/redo cycles and 6 fresh-process reopen checks** on B1 bundles in **152.5904 s**.
Maximum sampled concurrent Blender process-tree RSS was **1,244,901,376 bytes**
(sampled every 0.1 s). GLB geometry and appearance checks passed.

An independent fresh Blender process compares actual initial native face IDs,
owners, directed world-space triangle corners and linear RGBA against the
original source mesh. It passed all six scenes: maximum corner error
**6.1440e-7 m** and linear RGBA difference **2.1944e-7**, each below **1e-6**.
No nearest-point remapping is used in this source-lineage comparison.

The scripted runner reassigns fixed 12/8-face selections. It verifies data
preservation; it does not demonstrate useful boundary correction. The dining
download contains the **initial** B1 scene, 96,518 faces including 3,605 unresolved
faces, the standalone panel, complete B1 surface bundle and hash manifest.
Follow the [author task](../docs/OWNERSHIP-S2A.md) and record actual actions/time.

## Evidence, checks and limits

- [Diagnostic JSON](ownership-s2a/diagnostics-results.json) and [all-object area CSV](ownership-s2a/diagnostics-metrics.csv).
- [B1 export checks](ownership-s2a/partition-export-results.json), [native editing results](ownership-s2a/edit-suite-results.json), [independent lineage](ownership-s2a/native-lineage-results.json).
- [Figure PDF](ownership-s2a/ownership-failure-causes.pdf), [figure provenance](ownership-s2a/figure-manifest.json), [artifact manifest](ownership-s2a/manifest.json).
- [Reproduction commands](../docs/OWNERSHIP-S2A.md) and [Chinese instructions](../docs/OWNERSHIP-S2A.zh-CN.md). Per-face NPZ traces are regenerated by the public diagnostic command; their measured hashes are in the JSON. The public summary does not duplicate the roughly 30 MB of local raw traces.
- The original full CPU suite passed 119 tests. Additional public-package checks verify delivered bytes and the example archive. Portable CI does not run Blender; native results above were measured locally with Blender 5.1.2.

Measured environment: Windows 11, Python 3.12.14, NumPy 2.3.5, Open3D 0.19.0,
Blender 5.1.2, CPU. Figure export used Matplotlib 3.10.0 with NumPy 2.1.3 in a
separate existing environment. Its default threaded MKL path initially failed
to load a Windows DLL; setting `MKL_THREADING_LAYER=SEQUENTIAL` for that plotting
process succeeded. Numeric diagnostics were already complete in the CPU environment.

Nearest-reference matching still has no normal/visibility filter. Unreconstructed
surfaces are absent from the partition denominator. Six authored development
groups are not independent real-world or held-out validation. Human selection
quality, correction effort, whole-machine offline isolation and the planned
held-out study remain unmeasured. The result supports a focused next experiment
and a usable task starting point, rather than a claim of a new reconstruction method.

# Table0: cross-view diagnosis and a bounded joint-surface improvement

Measured 2026-10-15, Asia/Shanghai · [简体中文](TABLE0-ERROR-2026-10-15.zh-CN.md)

The same raw tea-room input now has a reproducible diagnosis and a joint-surface
rigid candidate that reduces Table0's mean on scoring observations excluded from estimation by **42.84%**.
It improves the target tail without the earlier scored background/Chair3
regressions. The original 80% support requirement and complete-neighbour gate
still fail, so it is retained as a diagnostic candidate. Original default
observations/cameras remain unchanged. A separate output bug is fixed: rejected
corrections can no longer silently become the ordinary scene used downstream.

## What the input can establish

The pinned [tea-room input](https://github.com/LiteReality/example-scans/tree/3f2ca5d618fe86b264716200baea19ae0aea0d49/tea_room)
remains 253 files, 10,518,452 bytes and 62 RGB/depth/confidence/camera sets.
All **639,790 observations regenerate bit-exactly**, with zero component or
colour error and maximum inverse pixel error 1.71e-13 pixels. Row-major poses,
Y/Z camera flips, axial millimetre depth and scaled intrinsics agree with the
[pinned upstream loader](https://github.com/LiteReality/LiteReality-Agent/blob/28ffe1fd4a1b03c93f1877a33a8bc9b806f72fe1/src/litereality_agent/pipeline/scene_init/ingest/preprocessing/vendor/litereality/scanner_capture.py).
Original and tested alternate conventions use the same training and validation
memberships; radial-range and half-pixel alternatives worsen the comparison.
There is no evidence for changing the input convention.

The measured discrepancy has two distinct components:

- **Near-view layer disagreement:** frames46–49 contain 6,169/7,127 validation
  observations (86.56%), with signed median normal residuals +15.42/+13.57/
  +16.65/+11.61mm. Shared5cm cells with frame4 retain same-sign differences,
  so simply observing a different tabletop region does not explain the shift.
- **Grazing/depth-boundary tail:** frames56–58 contain810 observations at median
  depths2.54–3.00m, incidence cosine .28–.32, and72–86% fixed raw-depth boundary
  association. They supply116/156 residuals above50mm and226/357 above the
  original P95. No observations are removed from the score.

Confidence2 is not a sufficient correctness signal:6,193 validation observations
have only19.47% support within10mm. Geometric box ambiguity is not a semantic
error label. The comparison plane also has uncertainty: training frame12 differs
from4 by11.18mm in shared-cell medians; leaving a major training view out shifts
its predictions. The unchanged plane is a comparison model, not surveyed truth.

These facts identify a view-dependent layer mismatch and a distinct long-tail
failure class. They do **not** uniquely identify physical sensor bias, pose
error, RGB/depth timing or motion. One saved timestamp cannot establish exposure
synchronization. Additional full-frame depth/ICP/colour diagnostics did not
produce an adoptable room-wide repair. Colour odometry's own success flag was
insufficient: two of four fixed pairs diverged beyond the independent bounds.

## The implemented improvement

The previous control estimated only translation along the tabletop normal. The
new candidate jointly fits the tabletop training plane and fixed training
neighbour/background point-to-plane constraints. Each nonempty group has unit
total weight; a10mm Huber loss is solved by six min-norm IRLS iterations using a
fixed small-motion Jacobian. Rotation is applied about the original camera
centre. Bounds remain80mm camera-centre translation,3° rotation, rank6 and a
1e-4 minimum normal-matrix eigenvalue ratio. Correspondences never rematch.

The original even-index, unique-owned, common-training-cell target quality gate
selects the four eligible later frames; all31 later-frame records are retained; the other27 keep
identity. Estimation uses later even indices; all3,558 odd scoring observations
remain disjoint but share frames and sensor. The original5,262 training observations and full7,127 later
validation observations are preserved. This development scan has already been
inspected; a separate protocol invocation freezes implementation/input hashes
and constants, not a claim of prospective independent testing.

| Same3,558 scoring observations | Mean | P95 | Within10mm |
|---|---:|---:|---:|
| Original fixed-plane comparison |15.393mm|26.303mm|22.990%|
| Prior normal translation |7.368mm|25.767mm|83.390%|
| Joint fixed-correspondence candidate |**8.799mm**|**25.471mm**|**75.295%**|

The prior control's higher support came with harmful neighbour/background
changes. The joint candidate passes available scored controls: Chair3 mean
6.609→6.849mm is within the original +2mm tolerance, its P95 is23.348→19.095mm;
background mean13.667→13.476mm and P9538.540→38.459mm, with all sufficiently
sampled per-frame controls passing. These are tolerated changes, not a claim
that every Chair3 residual decreases.

Full7,127 denominator (includes estimation samples): mean15.489→8.918mm,
P9526.409→26.355mm and support22.759→75.179%. The full-tail improvement is only
0.055mm; the long tail remains. **Overall FAIL:** target support is below80%,
Chair1 has no early training support, and frame46/47 Chair3 evidence is only
2/11 even estimation and3/7 odd scoring points, below the original30-point
per-frame score minimum. Missing evidence is preserved explicitly.

## A real output-contract defect and its fix

The previous rejected alignment directory still contained41,787 moved points
(max19.424mm) and four modified camera records under ordinary scene filenames.
The complete native delivery used its original input and was unaffected, but a
downstream caller could consume this rejected bundle. The TSDF path reloaded
raw camera poses for integration while exporting the modified metadata, yielding
inconsistent geometry/camera descriptions.

Now a failed gate writes bit-identical original `observations.npz`, original
camera records and `disposition=identity_fallback` as the ordinary output.
`candidate-observations.npz` and `candidate-scene.json` carry an explicit
diagnostic schema; the candidate scene points to its matching NPZ. Candidate points and cameras receive
the same rigid transform, preserving the original camera-space rays/depth.
The surface importer rejects diagnostic/legacy-rejected inputs and preflights
all frame poses against raw capture before integration or output. Its present
raw-RGBD path deliberately refuses unsupported derived camera metadata.

Regression tests reproduce the original defect before the fix. The local replay
then checks default identity, candidate projection consistency, byte-identical
original/fallback surfaces, and rejection before any candidate surface is made.


Final local replay:11 guarded stages complete, all253 source files unchanged,
zero runtime Python network events. Four candidate camera records change, with
maximum camera-space component discrepancy1.78e-15m; default original points and
cameras remain unchanged. Original/fallback TSDF bytes and12 part hashes match.

## Reproduce and inspect evidence

From this repository in an already prepared CPU Python environment:

```powershell
python -B reproduce_table0_diagnosis.py --input 'D:\LocalData\tea_room' --output '.local\table0-diagnosis'
```

Exit **2** means `completed=true` with `alignment_gate_passed=false`.
The driver does not download input, install packages, use a model/API or require
a GPU/Blender. Every stage uses the CPython DNS/socket guard; this is not OS-level
isolation of native code or subprocesses. The source manifest and transitive
implementation hashes are checked before/after execution. See the
[bilingual runbook](../examples/real-tearoom-local/TABLE0-DIAGNOSIS.md).

Numeric evidence: [input contract](table0-error/input-contract.json),
[residual diagnosis](table0-error/diagnosis.json),
[normal control](table0-error/normal-translation.json),
[joint candidate](table0-error/balanced-pose.json), [replay](table0-error/replay.json).
These exclude scan coordinates, camera matrices and rendered assets. Raw and
scan-derived material stays local under the existing source terms; the accepted
[synthetic editable scene](../examples/dining-complete/README.md) remains publicly
downloadable. No new paid resources or dependencies were used.

This is an engineering improvement to a documented failure, with a still-rejected
geometric candidate. Independent lawful capture with measured dimensions is
needed to evaluate physical accuracy and generalization. Next acquire or identify
independent pose/depth/timing evidence before adopting a room-wide transform;
do not equate a lower consistency residual with a unique causal diagnosis.

Method references checked2026-10-15: [Open3D ICP](https://www.open3d.org/docs/release/tutorial/pipelines/icp_registration.html),
[colour/hybrid RGB-D odometry](https://open3d.org/docs/latest/tutorial/pipelines/rgbd_odometry.html),
[Apple scene depth](https://developer.apple.com/documentation/arkit/ardepthdata/depthmap).

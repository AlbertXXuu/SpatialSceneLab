# Real tea-room: complete editable authoring and a failed geometric correction

Measured 2026-10-15, Asia/Shanghai · [中文](REAL-TEAROOM-2026-10-15.zh-CN.md)

This stage delivers a complete **local editable structural proxy scene** for
the actual 62-frame tea-room scan, plus reproducible code and evidence. It does
not establish accurate recovery of the complete real room. Table0's normal-offset
control improves its disjoint scoring observations, but fails neighbouring
chair/background checks. That correction is not adopted as accepted camera pose.
The native authoring workflow keeps the captured observations and cameras intact.

## Frozen input and scoring

The input remains the [pinned LiteReality tea-room scan](https://github.com/LiteReality/example-scans/tree/3f2ca5d618fe86b264716200baea19ae0aea0d49/tea_room):
253 files, 10,518,452 bytes, 62 RGB/depth/confidence/camera sets. The complete
[source manifest](../docs/tea-room-source-manifest.json) is checked before and
after the end-to-end replay. Missing, modified, duplicate, escaped or additional
source files are rejected. Original input and the earlier [failed baseline](REAL-TABLETOP-2026-10-09.md)
remain unchanged.

Table0 is `46C25048-47D3-45BE-BEDF-D3DF0DF758F6`. The original 5,262 training
points from frame <=30 determine the fixed plane. Every original 7,127 later-view
validation-band observation remains. Later-view estimates use only even-index
unique-owned observations in the fixed training-support cells; **3,558 odd-index
observations score the estimates, with zero estimation/scoring overlap**. This
is sample isolation within the same capture, not an independent sensor or final
test capture. RoomPlan boxes and unique ownership are priors, not semantic truth.

The new control estimates only translation along the fixed plane normal. It
does not estimate tangent translation, yaw or tilt rotations. A separate protocol
freezes code/input/control hashes, parameters and gates before evaluation. Failed
frames keep identity; neither frame selection nor thresholds use scoring outcomes.

| Same 3,558 disjoint scoring observations | Mean | P95 | Within 10 mm |
| --- | ---: | ---: | ---: |
| Fixed training plane | 15.393 mm | 26.303 mm | 22.990% |
| Frozen background ICP | 11.649 mm | 34.714 mm | 58.151% |
| Plane-normal offset control | 7.368 mm | 25.767 mm | 83.390% |

The original full 7,127 denominator includes estimation samples and is reported
separately: mean **15.489 -> 7.473 mm**, within-10-mm support **22.759% ->
83.275%**, but P95 **26.409 -> 26.515 mm**. The complete error tail remains
unresolved. No observations were dropped to improve these comparisons.

## Why the scene correction is rejected

The original 80%/10-mm target gate passes, but the required consistency controls
fail. Chair3's fixed training-match mean residual grows **6.609 -> 10.812 mm**.
Frame 48's background P95 worsens by **5.486 mm**, beyond the frozen 5-mm
worsening allowance. Chair1 lacks independent training support and fails closed.
The aggregate `passed` remains **false**. A coherent tabletop offset cannot
distinguish depth bias, camera error or scene motion, or justify moving the whole
frame. The diagnostics use actual fixed point-to-plane correspondences, following
the [Open3D official ICP formulation](https://www.open3d.org/docs/release/tutorial/pipelines/icp_registration.html)
(checked 2026-10-15).

[Public alignment evidence](real-tearoom/alignment-summary.json) retains counts,
fixed parameters, per-frame outcomes, neighbours, input/code hashes and failures.
Detailed transforms, corrected points and camera records remain local. An audit
CLI exit code **2** means geometric rejection after diagnostics are written.

## Complete editable scene and provenance

The scene contains **12 RoomPlan identity roots, 82 closed structural components
and all 62 captured camera records**. The roots cover Table0, Chair0–3, Sofa0,
Dishwasher0, Sink0 and Storage0–3. These category names retain the source priors;
they are not independently confirmed semantic labels.

Table0's upper plane uses the original training observations. Slab thickness,
leg arrangement and unseen supports are explicitly authored parameters. Chairs,
cabinets and appliances are structural proxies sized/placed from RoomPlan priors,
with recorded authored component choices. Their observation mismatch is reported
separately; this is not a faithful recovery claim. The original observed surface
is retained in a hidden comparison collection. Captured object transforms and
cameras are preserved; the rejected normal-offset bundle is not substituted.

The scene has selectable object roots and child parts, metre units and an
explicit Y-up to Blender Z-up conversion. Verification covers closedness,
structural contact, floor placement, actual component separation, edits/undo,
native saving/fresh reopening and GLB round trip. The intentional sink/cabinet
nesting is declared separately from unintended inter-object collision. Structural
acceptance and failed measured reconstruction are separate status fields.

The 3,075 cross-object component pairs include **3,040 separation checks and
35 explicitly declared Sink0/Storage2 nesting pairs**. They do not establish
zero intersection across every pair. The 24 authored feet have maximum floor
height gap **1 mm** and their 96 corner points lie inside the source Floor0
footprint; that floor is a RoomPlan prior, not independent physical ground truth.
GLB normals split vertex indices at seams. Its closedness check uses only exact
coordinate welding after one-to-one oriented-triangle/multiplicity comparison;
it does not assume glTF preserves the original welded vertex indexing.

[Structural evidence](real-tearoom/structural-summary.json),
[native evidence](real-tearoom/native-summary.json) and
[end-to-end replay evidence](real-tearoom/replay-summary.json) provide the final
measured counts, gates, tolerances and hashes. These are numerical receipts,
not redistributed scene geometry.

## Reproduce, open and edit

[The bilingual local runbook](../examples/real-tearoom-local/README.md) prepares
the pinned source, runs [reproduce_real_tearoom.py](../reproduce_real_tearoom.py),
opens the native scene and explains its comparison layers/provenance. The single
entry verifies input, reconstructs observations/TSDF, replays the old baseline,
freezes/runs the new control, authors structural parts and invokes Blender build
and fresh reopening. Its exit **2** can coexist with `completed=true`: materials
were constructed, while the measured alignment gate failed. Other failed stages
stop execution and retain logs. No runtime downloads or dependency installs occur.

Every Python and Blender stage runs under the existing outbound socket/DNS guard.
This verifies Python operations; native libraries and independent subprocesses
are outside its scope. The guarded rejection of the geometric control is retained,
along with its zero outbound network events. Preparation can use the network;
execution reads prepared local files and needs no model/API service or GPU.

The source scan has no redistribution license in either its pinned or current
repository tree as checked 2026-10-15. Under the existing [artifact terms](../THIRD_PARTY_NOTICES.md),
raw input, native/GLB files and rendered results stay local. The public delivery
contains original code, tests, numerical evidence and bilingual instructions.
The accepted [synthetic dining scene](../examples/dining-complete/README.md)
remains the directly downloadable public native example.

## Limits and the next useful measurement

The local authoring/editing delivery is distinct from an accepted complete
observed reconstruction. The raw fixed-plane real tabletop still has P95
**26.409 mm** and only **22.759%** within 10 mm. A room proxy that looks complete
does not fix this failure, recover unseen topology, validate physical dimensions,
or establish semantic accuracy. This capture is already development data.

Next investigate the actual cause of the target/neighbor disagreement with
independent depth or pose evidence; further forced rigid correction is unsupported.
For generalization and physical accuracy, obtain an independent lawful capture
with measured dimensions. No new paid allocation or dependency installation was
used in this delivery.

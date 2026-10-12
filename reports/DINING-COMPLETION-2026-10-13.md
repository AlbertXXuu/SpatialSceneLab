# Complete editable dining scene: measured shape, layout and export

Measured 2026-10-13, Beijing time · [中文](DINING-COMPLETION-2026-10-13.zh-CN.md) ·
[Open, edit and reproduce](../examples/dining-complete/README.md)

The existing public dining development scene now contains the accepted stool,
a completed table and two completed chairs, with four independently selectable
roots and 37 editable furniture parts. The three new assemblies pass the fixed
original-observation triangle gate, retain their source evidence, and survive
native edits, undo/redo, fresh-process reopening and GLB export/import. Following
the user's discovery of a chair/table intersection, the final scene also passes
an independent separation gate for every one of its 507 cross-furniture part
pairs. The smallest certified native/export gap is approximately **5.549958 mm**.

These results establish a reproducible engineering stage: turning incomplete
observations into documented, editable assemblies and checking their usable
scene layout and export fidelity. A better-looking stool is a useful inspection
result; the stronger evidence is the measured error, preserved inputs, rejected
controls and replayable edit/export checks. This controlled synthetic development
case does not establish accuracy for arbitrary real furniture. In particular,
the fitted table's reverse-area reference P95 worsens from **17.933 to 48.378 mm**,
despite much better agreement with captured observations.

![Final authored dining layout with the original observed environment](../examples/dining-complete/native/dining-complete.png)

## Inputs, fitting and unchanged acceptance criteria

The input is the existing 18-view public synthetic dining capture, using
[scene.json](../examples/stool-fit/input/scene.json) and
[observations.npz](../examples/stool-fit/input/observations.npz). The operator
selects rectangular table and seat/back chair families with four straight round
supports. Geometry, colour and evidence selection use only the nine even frames
0–16, captured points/RGB, supplied candidate/owner records, upright object
priors and the measured floor. The nine odd frames 1–17 are development
validation. Their results informed method investigation, so they are not an
untouched final test set.

Competing prior boxes can assign a visible chair support to the table. The fitter
therefore uses even-frame target candidates to recover support evidence, while
using local original-owner even points for chair seat caps to avoid table
contamination. The independent evaluator reconstructs its candidate-even support
pool from the input records rather than trusting fitter support flags. Pool sizes
are 4,801 / 3,005 / 3,321 for table / west / east; original owner-even counts are
2,808 / 2,008 / 2,325. This training selection never changes the validation
denominator: all **2,840 / 1,848 / 2,451 original owner-assigned odd points** remain
in the respective scores, with **zero exclusions**.

The declared shape gate requires, separately for each new object, actual exported
triangle P95 ≤10 mm and at least 95% of those original odd observations within
10 mm. Every component must have measured candidate-even support, match its
parameters and assembled mesh, be closed with consistent outward winding and
positive volume, participate in a connected intended-contact graph, and have
its four support feet within 3 mm of the measured floor. Closed hidden faces and
continuation to contacts are declared structural completion. The parts retain
independent shells, including intentional overlap at joints; they are not a
destructively welded solid union.

[fit_dining.py](../fit_dining.py) records its input and source-code hashes and read
scope. [evaluate_dining.py](../evaluate_dining.py) separately measures actual PLY
triangles using Open3D 0.19.0. The physical reference pack is used only by this
independent evaluation after freezing; it is not a fitting input. The evaluator's
input/candidate before-and-after hashes are unchanged. The fitter's recorded
read scope is a declaration backed by inspectable code and replay, rather than
an OS-level record of every upstream file access.

## Failed controls and the accepted surface model

Training projections show crowned seats and bowed backs together with planar
base surfaces. A rectangular slab control fails, and a single domed slab still
fails the same gate. The final family uses rectangular seat/back bases with
separate closed convex half-ellipsoid cushions. Attempts with contaminated
candidate seat-cap pools also fail; the accepted attempt uses localized
original-owner even observations for the cap. Neither the fixed 10 mm / 95%
criteria nor the odd-point population was relaxed.

The table below uses **actual exported-triangle** validation for the captured
rectangular/domed controls and final geometry. Analytic residuals in the method
history are a different diagnostic and must not be substituted for these values.

| Surface attempt | West P95, mm | West within 10 mm | East P95, mm | East within 10 mm | Shape decision |
|---|---:|---:|---:|---:|---|
| Rectangular slabs | 21.926 | 71.970% | 19.939 | 70.828% | fail |
| Single domed slabs | 12.424 | 92.154% | 12.849 | 92.697% | fail |
| Separate bases/cushions, final owned cap selection | 5.668 | 98.593% | 7.570 | 97.103% | pass |

[method-history.json](dining-completion/method-history.json) retains five fitting
attempts and the frozen final run. [method-controls.zip](../examples/dining-complete/method-controls.zip)
contains their historical parameters/PLYs and two captured observation-only
evaluations; [method-controls.json](dining-completion/method-controls.json) records
the archive hash. Old paths and code hashes remain historical provenance. The
archive can be extracted locally and its meshes reevaluated with explicit paths.
[candidate-training-projections.png](dining-completion/candidate-training-projections.png)
records the observation-driven surface diagnosis. Export winding and parameter
correspondence are checked against the actual meshes, including cushion cap
planes and radii; imported Blender geometry is not regenerated from metadata.

## Original-frame observation and reference measurements

The original B1 meshes and final fitted meshes are evaluated against the same
frozen source-frame observations. The final gate passes for all three objects.
Results below come from [evaluation.json](dining-completion/evaluation.json);
distances are to the actual exported triangles.

| Object | Original odd points | B1 mean / P95, mm | B1 within 10 mm | Final mean / P95, mm | Final within 10 mm |
|---|---:|---:|---:|---:|---:|
| Table | 2,840 | 7.808 / 26.693 | 74.754% | 0.970 / 3.692 | 99.894% |
| West chair | 1,848 | 10.441 / 25.213 | 56.494% | 1.863 / 5.668 | 98.593% |
| East chair | 2,451 | 13.343 / 33.906 | 50.347% | 2.049 / 7.570 | 97.103% |

Every one of the 30 new parts has actual-triangle support within 10 mm; the
smallest counts are 79 for the table, 47 for west and 57 for east. All 30 read
back as watertight, orientable, positive-volume meshes in
[final-open3d-readback.json](dining-completion/final-open3d-readback.json). The
table has 10 certified intended contacts and each chair 15. All three contact
graphs are connected and their four feet pass the 3 mm floor gate. The largest
measured foot gap is 0.000469 mm in these synthetic inputs.

Independent physical-reference diagnostics show both improvement and remaining
shape error. Captured-ray recall counts repeated valid rays from every supplied
view. Reverse accuracy uses 30,000 uniform triangle-area samples per mesh, seed
20261013, including component-shell faces inside overlaps.

| Object | Reference rays | Ray recall ≤10 mm, B1 → final | Ray P95, mm, B1 → final | Reverse-area P95, mm, B1 → final | Reverse-area ≤10 mm, B1 → final |
|---|---:|---:|---:|---:|---:|
| Table | 31,128 | 55.705% → 96.887% | 224.537 → 6.450 | **17.933 → 48.378** | **83.307% → 57.970%** |
| West chair | 19,338 | 46.080% → 94.467% | 263.836 → 10.493 | 27.120 → 25.088 | 59.830% → 74.220% |
| East chair | 23,292 | 41.473% → 92.972% | 260.804 → 12.204 | 35.599 → 23.502 | 54.030% → 71.883% |

These are distinct populations from owner-assigned odd validation. Ray recall
does not estimate full or visible surface-area completeness; reverse shell
sampling does not measure exposed union area. Neither diagnostic is the
declared observation gate. The table's fitted top thickness is 118.415 mm,
versus 70 mm in the evaluation-only reference; its lower face accounts for a
48.378 mm offset. Its fourth leg also has reverse component P95 69.268 mm.
The strong observed-surface result therefore coexists with substantial completed
shape error. No fitting or geometry change was made in response to these frozen
reference results.

## User correction: independent cross-furniture separation

The first native acceptance checked joints inside each furniture object and
export fidelity, but omitted collisions between different furniture. The user
identified a chair seat intersecting a table leg. Independent diagnosis of the
actual frozen triangles then found **10 positive-solid collision pairs** and
one additional pair with only **3.153 mm** certified clearance. The accepted
stool had no collisions. Earlier native renders and pass records remain local
historical evidence, not the final complete-scene delivery.

[evaluate_dining_layout.py](../evaluate_dining_layout.py) verifies 37 actual
closed convex component meshes, detects common interior witnesses and uses
closest-simplex convex distance for crossing solids. Final acceptance relies
on vertex-bound separating planes: every vertex of the first component lies
on one side, every vertex of the second on the other, and their unit-normal
projection gap exceeds the required separation. This also separates their
triangles and enclosed solids. The certificates cover all 507 distinct-object
part pairs, without replacing actual meshes by guessed dimensions or reference
furniture. [layout.json](dining-completion/layout.json) preserves the original
failures, final certificates, frozen inputs and canonical transform hash.

Only explicit horizontal root translations author the final layout. The table
and stool are fixed. A bounded outward search along each existing chair axis
uses a 5.5 mm certification target plus authoring precision margin; it is not an
unrestricted layout optimization.

| Root | Authored translation in world Y-up metres [X, Y, Z] |
|---|---|
| Table | [0, 0, 0] |
| West chair | [−0.1118755615, 0, approximately 0] |
| East chair | [+0.1049262697, 0, −0.0147464255] |
| Accepted stool | [0, 0, 0] |

The source-frame fit/observation/reference scores above remain unchanged. They
are **not scores of the translated final display layout**. All PLY bytes and
original observed object transforms remain in their original frame; root
translations are stored separately. The independent minimum certified gap is
5.550033615 mm. Blender rechecks the same planes against its actual imported
world vertices before saving, after fresh reopening, after edit undo, and after
GLB import: native/undo minima are **5.549958020 mm**, GLB **5.549958393 mm**, all
above the required 5 mm. The native record binds the public layout JSON and its
canonical translations to the exact fitted and stool inputs.

## Editable native delivery and evidence retention

[complete-dining.blend](../examples/dining-complete/native/complete-dining.blend)
is the editable authority; [complete-dining.glb](../examples/dining-complete/native/complete-dining.glb)
exports visible geometry, hierarchy and metadata. The independently selectable
roots are `Fitted Stool (structural completion)`, `Completed Table`,
`Completed Chair West` and `Completed Chair East`.

| Furniture | Editable parts | Triangles |
|---|---:|---:|
| Existing accepted stool | 7 | 2,240 |
| Completed table | 8 | 2,252 |
| Completed west chair | 11 | 14,552 |
| Completed east chair | 11 | 14,552 |
| Furniture total | 37 | 33,596 |

The table has a top, four legs and three rails. Each chair has a seat/base,
four legs, seat cushion, back/base, back cushion, two back posts and a back rail.
Root Transform controls move/rotate/resize an assembly; child Transform/Dimensions
and native Edit Mode allow component edits. Parameters and source hashes record
the accepted fitting state; they do not automatically regenerate meshes.

All **17 preexisting source objects** retain exact geometry, transforms,
attributes, materials and properties. Observed table/chairs and `Unresolved
(-2)` move only between collections/visibility states, into the switchable
`Original dining observations + ambiguous fragments (hidden)` layer. The
ambiguous object retains identity `unresolved`, 1,996 vertices and 3,605 triangles,
its recorded bounds, and visibility in the before render. It is recorded as
ambiguous observed dining fragments. The original environment remains visible;
the accepted stool stays unchanged. The original observed stool has its own
comparison layer.

Each new furniture root passes a scripted native translation [0.15, 0.10, 0] m
in Blender coordinates, 17° Z rotation and 1.05 X scale, with coherent child
movement and unrelated meshes unchanged. Undo, redo and a second undo restore
the authored layout and all preserved sources. A fresh Blender 5.1.2 process
reopens the saved file. Ten saved 3D viewports have helper overlays hidden.
The visible GLB contains **38 meshes**, including the original environment.

| Fidelity check | Maximum error |
|---|---:|
| New native vertices versus frozen PLY plus authored translation; face indices exact | 0.230463 µm |
| Native root affine edits versus expected child geometry | 0.186002 µm |
| GLB round-trip oriented triangle corner match | 1.177270 µm |

The native record additionally verifies all seven accepted stool components
against their original PLY geometry. GLB checks include visible inventory,
root/part hierarchy, source PLY provenance and original environment appearance.
Arbitrary Blender per-face lineage remains a native-file property.
[scene-integration.json](../examples/dining-complete/native/scene-integration.json)
contains these actual checks and the repeated collision gate.

The following are actual Blender renders, with unchanged cameras between the
historical and final runs. The before/complete pair uses exactly the same camera
and lighting; before shows retained observed fragments and the accepted stool.
All four final views were visually inspected after the layout correction.
The final metadata-bound replay produces exactly the same decoded RGBA pixels
as those reviewed images; PNG bytes differ because Date/RenderTime metadata changes.

![Same-camera original observation comparison](../examples/dining-complete/native/dining-before.png)

![Front view of the final authored furniture layout](../examples/dining-complete/native/dining-front.png)

![Underside view of supports and rails](../examples/dining-complete/native/dining-underside.png)

## Replay, guards and verification

The measured host is Windows x64. Fitting uses CPython 3.13.5, NumPy 2.1.3 and
SciPy 1.15.3 ([requirements-fit.txt](../requirements-fit.txt)); independent
geometry checks use CPython 3.12.14, NumPy 2.3.5 and Open3D 0.19.0
([requirements-local.txt](../requirements-local.txt)). Native validation and
the actual rendered views use Blender 5.1.2. These are separate existing
environments, rather than one environment with every optional dependency.

[The bilingual runbook](../examples/dining-complete/README.md) provides existing
Windows interpreter paths, dependencies and separate fit → shape evaluation →
layout evaluation → native build → fresh reopen commands. Outputs must be new;
the importer rejects stale evaluations, non-horizontal translations, changed
inputs, unsafe PLY paths, certificate omissions and destructive destinations.
The existing Y-up to Blender Z-up rotation is applied once at PLY import; no
new geometry-fitting UI is needed.

Independent fitting replay reproduces all **33 component/assembly PLY files
byte-for-byte** in [independent-fit-replay.json](dining-completion/independent-fit-replay.json).
The [shape evaluator verification](dining-completion/evaluator-verification.json)
records byte-identical reference re-extraction. The final
[Open3D-environment suite](dining-completion/final-tests.log) runs **240 tests,
zero failures/errors, four skips** (optional SciPy fitting dependency), including
**22 shape, 12 layout and 18 native-input/separation tests**, all passing.
The final [conda fit-environment suite](dining-completion/final-fit-environment-tests.log)
also runs 240 tests with zero failures/errors and 106 skips from absent Open3D;
all 11 dining fitter tests pass there. These existing environments provide
complementary coverage. An earlier pre-layout conda run had 218 tests and 92
skips; it remains historical partial coverage.

Post-review controls bind the capture digests before reading, verify them after
reading and again after fitting, and refuse a result if inputs change. Both
independent evaluation CLIs now return failure exit codes for failed gates.
The final fit/evaluation/layout metadata chain was regenerated, followed by a
new native build/reopen. All 33 PLY files, shape results, layout transforms and
507 certificates remain unchanged; this strengthens acceptance execution and
input binding without changing furniture geometry.

Guarded fitting replay took 1.736 s; final native build 7.150 s and fresh reopen
17.814 s on this machine. These are individual recorded executions, not a
throughput benchmark. The guards report zero runtime Python network attempts
and successful DNS denial probes. The separate final layout guard also denied
reads from both physical-reference roots, with zero attempted reads. See
[fit guard](dining-completion/independent-fit-guard.json),
[layout guard](dining-completion/final-layout-guard.json),
[build guard](dining-completion/native-delivery-build-guard.json) and
[reopen guard](dining-completion/native-delivery-reopen-guard.json).
These checks cover Python socket/DNS/file-access operations as specified;
they do not establish OS-level isolation of native libraries or child processes.

| Final frozen artifact | SHA256 |
|---|---|
| Fit JSON | `1cb164fbe6272e48ca41507fd71e037aa70956131630d589e039c56d16c099a1` |
| Independent layout JSON | `0ddbf52fa39f65014bb199e819f3cc63a60adc1f06cf726fcfc6f94060685345` |
| Accepted source Blender scene | `27144f490ac987a6445acc46ef7ec1ceb1c89b87c8f057a057ede3f650617e2b` |
| Final Blender scene | `4dba3e5be30ed597d1fd5ca6e36449106e3eab9f1d3e8873e4437ace0ca507da` |
| Final GLB | `768a4c69b5fdc9e110be8b4031928aa7bd3d6a55615eb3b1a9c3cd13133dbbd2` |

The meaningful completed capability is now inspectable: observation support,
bounded shape acceptance, retained ambiguity, usable collision-checked layout,
native editing and faithful interchange. Real-capture completion, independently
held-out furniture and measured human correction effort remain separate evidence
needs. The [existing real-table report](REAL-TABLETOP-2026-10-09.md) still records its cross-view failure;
this synthetic completion does not resolve that result.

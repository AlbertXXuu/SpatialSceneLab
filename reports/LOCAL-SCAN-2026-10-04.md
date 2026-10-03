# Local RGB-D surface editing and Blender export validation

2026-10-04 · [中文报告](LOCAL-SCAN-2026-10-04.zh-CN.md) ·
[Machine-readable evidence](local-scan-results.json) ·
[Reproduction instructions](../docs/REPRODUCING.md)

**Result:** one real 62-frame RGB-D scan was reconstructed locally on Windows,
partitioned into measured surface regions, edited, saved in Blender, exported
to GLB and numerically reopened. The selected Sofa0 region moved by 0.3 m on X,
while the other mesh parts retained their world geometry. The engineering
checks passed within their stated scope. Complete semantic object extraction
and whole-machine network isolation remain unverified.

## Problem and acceptance scope

The 2026-10-04 task asks for useful, reproducible scene engineering on the
available Windows computer with 64 GB RAM, an RTX 4060 Laptop 8 GB and zero
additional paid-service budget. The concrete failure to avoid is an apparent
object edit that moves a RoomPlan proxy while its measured surface stays at the
old location, or an export that silently changes units, axes, identities or
unrelated objects.

The measured gate therefore checks calibrated RGB-D projection, provenance for
selected observations, an actual surface translation, unchanged other geometry,
Blender/GLB reopening and an original public control fixture. Known capture
poses and depth are inputs; this stage does not estimate a new camera trajectory.
The earlier controlled-cuboid and MapAnything/TUM coordinate experiment remains
documented in [Stage 1](STAGE-1.md). ScaRF is a historical optional candidate;
its environment gate is not a prerequisite for this local RGB-D entry.

## Environment and fixed sources

| Component | Measured configuration |
| --- | --- |
| Host | Windows x64, 64 GB RAM; RTX 4060 Laptop GPU inventory |
| Reconstruction interpreter | Python 3.12.14 |
| Direct Python dependencies | NumPy 2.3.5, Pillow 12.3.0, Open3D 0.19.0 |
| Scene authoring | Blender 5.1.2, build `ec6e62d40fa9` |
| TSDF / render device | CPU / CPU |
| New paid services | None used |
| Timing scope | One local run per reported point/TSDF timing; no inference or downloads |
| Peak RAM / GPU benchmark | Not measured; no CUDA model was run |

[Direct dependencies](../requirements-local.txt) are version pinned; the
[59-package environment lock](../requirements-local-lock.txt) records the
measured Windows x64 / Python 3.12 dependency closure.
Blender was obtained from the [official Blender 5.1 release directory](https://download.blender.org/release/Blender5.1/)
and its ZIP matched the official checksum:
`345bedea7b0acf7cc9666423d8553f9129622aea34ded65c23e8cb70f83f14ff`.
The Blender Python runtime is separate from the reconstruction environment.

The real input is [LiteReality/example-scans `tea_room`](https://github.com/LiteReality/example-scans/tree/3f2ca5d618fe86b264716200baea19ae0aea0d49/tea_room),
fixed at `3f2ca5d618fe86b264716200baea19ae0aea0d49`. It contains 253 files and
10,518,452 bytes: 62 RGB images at 1920×1440, 62 uint16 depth images at 256×192,
62 confidence images and 62 camera JSON records, plus point cloud and RoomPlan
files. Each file was checked against upstream Git blob identity/size and a
local SHA256 manifest; image decoding and both USDZ ZIPs passed. The
[public source manifest](../docs/tea-room-source-manifest.json) retains file
names, sizes and SHA256 values.

The pinned example-scans tree has no license file. The public resource includes
implementation, original MIT fixtures, source links, hashes and text metrics.
Real scan inputs, derived PLY/GLB/Blender assets and their rendered images stay
local. [THIRD_PARTY_NOTICES](../THIRD_PARTY_NOTICES.md) records this boundary and
the historical data/model attribution.

## Implementation

[scan_pipeline.py](../scan_pipeline.py) loads the scanner's row-major camera
records, scales RGB intrinsics to depth resolution and changes the ARKit camera
basis from -Z forward/+Y up to the RGB-D +Z forward/+Y down basis. It preserves
the Y-up world in meters. This follows the pinned upstream
[scanner format reader](https://github.com/LiteReality/LiteReality-Agent/blob/28ffe1fd4a1b03c93f1877a33a8bc9b806f72fe1/src/litereality_agent/pipeline/scene_init/ingest/preprocessing/vendor/litereality/scanner_capture.py#L580-L587).
The new parser accepts one static parametric RoomPlan profile, including
row-vector USD matrices, categories, UUIDs and authored dimensions; unsupported
transforms or layers fail explicitly. The source label `Door(Isopen: True)` is
retained while its geometric category is normalized to Door.

The point entry filters zero depth, the 65535 sentinel, low confidence and
depth beyond 6 m. It reads depth in millimeters with scale 1000, samples every
two depth pixels, and bilinearly resizes RGB to depth resolution. Each retained
measurement keeps its color, frame index and integer depth-pixel UV. RoomPlan
oriented boxes with a 0.025 m margin supply a geometric ownership prior;
finite floor/wall/door/window regions use 0.025 m exclusion tolerance. Unique
candidates become object regions, overlapping candidates stay ambiguous, and
other points remain environmental observations. PLY export selects one real
point per 0.015 m voxel within each layer; the full retained arrays remain in
`observations.npz`.

[reconstruct_mesh.py](../reconstruct_mesh.py) reuses Open3D's CPU
`ScalableTSDFVolume` with 0.035 m voxels and 0.105 m signed-distance truncation.
It integrates the full valid 256×192 depth images for the selected 62 frames,
using known world-to-camera extrinsics. A mesh face becomes editable only if
all three vertices have one identical object owner. The remaining faces stay
in the environment and are counted, rather than being assigned arbitrarily.
All output parts receive hashes and retain their RoomPlan UUIDs.

The observation scene records SHA256 for every used RGB/depth/confidence/camera
file, RoomPlan USDZ and reference PCD. The surface entry checks those source
hashes again, rejects resolved path escape, and checks output-write success.
Source drift, empty/non-finite reference data, invalid USD indices and edited
metadata corruption are covered by rejection tests.

[blender_scene.py](../blender_scene.py) imports the hashed surface parts using
native PLY I/O, applies the world conversion `(X,Y,Z) → (X,-Z,Y)` into Blender's
Z-up meter scene, and preserves identities in custom properties. It translates
the selected measured mesh region, renders before/after from one fixed review
camera, saves both `.blend` scenes and exports both GLBs using native glTF I/O.
The verification imports each GLB into a fresh scene and reopens the edited
`.blend`; it compares UUIDs, roles, triangle counts/connectivity, source hashes,
physical world vertices and vertex-color/material presence.
Vertex comparisons are bidirectional nearest-neighbor distances, allowing
normal/material vertex splitting during export. The tolerance is 0.00005 m.
Capture camera nodes retain poses and source-intrinsic metadata; their viewport
lens is illustrative rather than a complete calibrated optical projection.

## Observed results

| Check | Actual result |
| --- | --- |
| Automated source/scan/network controls | 24 passed in 1.092 s; physical projection, ownership, edit invariants, source integrity and guard checks |
| Real scan observations | 639,790 valid sampled points from 62 frames |
| RoomPlan metadata | 12 objects, 14 structural components |
| Unresolved overlapping point candidates | 11,944 observations |
| Inverse projection | Max pixel error 1.7053025658242404e-13 px; max depth error 1.7763568394002505e-15 m |
| Point reconstruction wall time | 3.82861560001038 s, one CPU run including source hashing |
| Sofa0 point edit | 37,655 selected observations moved +0.3 m X |
| Saved/reloaded point translation error | 0 m maximum absolute error |
| Unselected points / other object metadata | Exactly unchanged |
| Point colors, labels and provenance | Exactly unchanged |
| Selected point IDs retained in environment | 0 |
| Unresolved Sofa0 point candidates | 0 under the geometric prior |
| TSDF surface | 83,425 vertices, 155,768 triangles |
| TSDF reconstruction wall time | 2.870004099997459 s, one CPU run including source validation |
| Surface partition | 12 mesh parts: environment + 11 observed object regions; all triangles accounted for |
| Environment / ambiguous mesh vertices | 135,156 environment triangles / 1,562 ambiguous vertices |
| Same-scanner PCD comparison | 73,651 reference points; median vertex distance 0.02800760519847716 m, p95 0.04497257413611268 m |
| Sofa0 fused editable region | 2,480 vertices, 4,409 triangles |
| Sofa0 candidate boundary faces kept in environment | 275 triangles |
| Blender target translation | +0.3 m X in source Y-up meters |
| Blender edit world-vertex error | 1.1920928955078125e-7 m maximum |
| Original and edited GLB reimports | Each maximum world-vertex distance 8.940696716308594e-7 m |
| Edited `.blend` reopened | 0 m maximum world-vertex distance |
| Source hashes / triangle connectivity | Retained through the edit and all native round trips |
| Vertex colors / materials | Present after native round trips; quantitative appearance fidelity unmeasured |
| Blender injected failure controls | All 5 detected: missing identity, unrelated vertex change, changed source hash, missing vertex colors, changed triangle connectivity |

Sofa0's source UUID is `4DF84D78-0A18-4140-86A7-2DF1360A4AE2`.
The point and mesh edits both address this identity, but they edit different
representations: selected raw measurements and selected fused surface faces.
Point-label exclusivity proves that selected measurements are not duplicated
in the environment. It does not prove semantic instance completeness. The mesh
retains 275 Sofa0 candidate boundary triangles, so the demonstrated Blender
edit is explicitly a selected surface-region move.

The original public fixture independently ray casts two physical boxes, a floor
and a back wall from two cameras, with semantic labels and at most 0.0005 m
depth quantization error. Its tests check physical coordinate expectations,
intrinsic scaling, proper poses, source labels and saved point edits. Deliberate
wrong units/axes, a transposed pose, invalid depths, proxy-only motion and an
unrelated-object change are rejected. The published fixture uses
`--resolution-scale 4`, producing two 192×128 depth images, 384×256 RGB and
12,284 sampled observations. Its full surface chain produced 9,081 vertices,
16,820 triangles and three Blender mesh parts, then passed
the native save/export/reopen checks after a +0.3 m X edit. The
[public fixture bundle](../examples/local-scan-fixture/) supplies reusable
original render and GLB evidence under MIT.

![Original public fixture: before and after the selected-region edit](../examples/local-scan-fixture/edit-comparison.png)

## Network preparation and guarded execution

Dependency installation and input preparation happened online. The subsequent
point reconstruction, TSDF and Blender execution used already prepared local
files. A Python audit guard rejected a deliberate DNS request and blocked
Python outbound socket/DNS operations; the reconstruction workload completed
without rejected network attempts. All seven retained guard reports passed,
including the selected-point edit and original fixture runs. Local socket binding was allowed because
urllib3 performs an IPv6 capability probe by binding `::1` locally.

This is a process-level Python network audit, installed separately in the
reconstruction interpreter and Blender's embedded Python. Native C networking
and independent child processes are outside its coverage.
It establishes the recorded guarded Python execution, while a whole-machine
disconnected rerun is still pending. The [runbook](../docs/REPRODUCING.md)
separates online preparation, local execution and the user's isolation check.

## Interpretation and remaining work

The results establish coordinate and unit handling, provenance preservation,
selected observed-point and surface edits, and native Blender/glTF world
geometry retention on this machine. TSDF and Blender are mature components;
the contribution is their measured integration with a constrained scanner
parser, explicit object priors, original controls and reproducible validation.
No new reconstruction algorithm or paper-level novelty is claimed.

There is one real scan and no independent real-world dimension or instance
ground truth. Numerical inverse projection and the same scanner's PCD do not
measure physical accuracy. RoomPlan boxes can include the wrong surface or miss
thin/occluded geometry, and low-resolution depth plus 0.035 m TSDF voxels limit
detail. Fused geometry represents observed surfaces; unobserved backs, complete
furniture assets, collision-ready solids, material recovery and arbitrary RGB
video input have not been demonstrated. Quantitative appearance fidelity and
whole-machine network isolation require their own evidence beyond the geometric
checks reported here.

For an engineering portfolio, the defensible claim is: **built and tested a
local RGB-D scene pipeline with traceable observations, object-region edits and
native Blender/GLB round-trip validation on Windows**. The source, negative
controls, machine results, input hashes, bilingual runbook and original public
fixture make that claim inspectable. The next gate is to resolve the retained
object-boundary regions and establish semantic selection quality, then rerun
the prepared chain under recorded whole-machine network isolation.

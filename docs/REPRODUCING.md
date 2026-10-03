# Reproducing the local RGB-D experiment

[中文说明](REPRODUCING.zh-CN.md) · [Measured report](../reports/LOCAL-SCAN-2026-10-04.md)

Run the commands below from the SpatialSceneLab repository root in PowerShell.
The supported input for this experiment is a prepared ScannerApp RGB-D folder:
`frame_*.jpg`, `depth_*.png`, `conf_*.png`, `frame_*.json`, a parametric
`room.usdz` and `pointcloud.pcd`. The scene hashes these used files and the
surface entry revalidates them before execution. The parser accepts the documented static RoomPlan profile;
general USD assets use Blender's native importer.

The report records the actual tested environment and measurements. Preparing
software, wheels and source input can use the network. Reconstruction, editing,
rendering and validation then read local files.

## Prepare software and dependencies once

Use CPython 3.12 with `venv` and `pip`; the measured run used Python 3.12.14 and
Windows x64 Blender 5.1.2, build `ec6e62d40fa9`. Obtain Python from [python.org](https://www.python.org/downloads/)
and Blender from its [official release archive](https://download.blender.org/release/).
The commands use explicit executable paths so they work without changing PATH
or the shared Python environment.

```powershell
$Python312 = 'C:\path\to\Python312\python.exe'
& $Python312 -m venv '.local\venv312'
$Python = (Resolve-Path '.local\venv312\Scripts\python.exe').Path
& $Python -m pip download --only-binary=:all: -r requirements-local-lock.txt --dest '.local\wheelhouse'
& $Python -m pip install --no-index --find-links '.local\wheelhouse' -r requirements-local-lock.txt
& $Python -m pip freeze | Set-Content -Encoding utf8 '.local\installed-packages.txt'
Get-ChildItem -LiteralPath '.local\wheelhouse' -File | Get-FileHash -Algorithm SHA256 | Export-Csv -NoTypeInformation -Encoding utf8 '.local\wheel-sha256.csv'
$Blender = 'C:\path\to\blender\blender.exe'
& $Python --version
& $Blender --version
```

Keep the Python installer, Blender archive, wheelhouse, wheel hashes and source
checkout together with your reproduction materials. Creating another Python
3.12 environment and repeating the `pip install --no-index` command tests the
prepared wheelhouse without an index connection. [requirements-local-lock.txt](../requirements-local-lock.txt)
records the measured 59-package dependency closure for Windows x64 / Python
3.12. [requirements-local.txt](../requirements-local.txt) pins the three direct
dependencies. Retain `installed-packages.txt` to compare your installed closure;
the provided Windows lock does not establish support for another platform.

The measured Blender ZIP SHA256 is
`345bedea7b0acf7cc9666423d8553f9129622aea34ded65c23e8cb70f83f14ff`;
it was checked against the official Blender 5.1 download directory's checksum.
Blender carries its own Python runtime, separately from the reconstruction venv.

## Prepare the pinned real input

The measured input is `tea_room` at LiteReality/example-scans commit
`3f2ca5d618fe86b264716200baea19ae0aea0d49`. The pinned repository has no license
file. Obtain the input from its [upstream source](https://github.com/LiteReality/example-scans/tree/3f2ca5d618fe86b264716200baea19ae0aea0d49/tea_room)
under the applicable permissions. Original input and all scan-derived geometry,
Blender scenes and renders stay local; see [artifact terms](../THIRD_PARTY_NOTICES.md).

With Git already available, the following prepares only the selected folder:

```powershell
git clone --filter=blob:none --no-checkout https://github.com/LiteReality/example-scans.git '.local\example-scans'
git -C '.local\example-scans' sparse-checkout init --cone
git -C '.local\example-scans' sparse-checkout set tea_room
git -C '.local\example-scans' checkout --detach 3f2ca5d618fe86b264716200baea19ae0aea0d49
$Scan = (Resolve-Path '.local\example-scans\tea_room').Path
```

The measured inventory is 253 files and 10,518,452 bytes, with 62 matching RGB,
depth, confidence and camera records. Git's pinned checkout identifies the
source content. For audit retention, also record file SHA256 values before
running. The [source manifest](tea-room-source-manifest.json) lists the measured
253 file sizes and SHA256 values. A shallow or partial clone must finish fetching the selected files before
disconnecting; none of the execution commands performs a Git checkout.

Verify all prepared source files against the public SHA256 manifest:

```powershell
$InputManifest = Get-Content -LiteralPath 'docs\tea-room-source-manifest.json' -Raw | ConvertFrom-Json
foreach ($ScanEntry in $InputManifest.files) {
    $ScanFile = Join-Path $Scan ($ScanEntry.path -replace '^tea_room/', '')
    $ActualHash = (Get-FileHash -LiteralPath $ScanFile -Algorithm SHA256).Hash.ToLowerInvariant()
    if ((Get-Item -LiteralPath $ScanFile).Length -ne $ScanEntry.bytes -or $ActualHash -ne $ScanEntry.sha256) {
        throw "Input checksum or size mismatch: $($ScanEntry.path)"
    }
}
```

## Run the public synthetic control

The original MIT fixture ray casts two physical boxes, a floor and a back wall
from two cameras. It includes independent semantic labels and millimeter depth
quantization. It can be generated and edited locally with no external input:

```powershell
& $Python fixtures/make_rgbd_fixture.py --output '.local\fixture-input' --resolution-scale 4
& $Python scan_pipeline.py reconstruct --input '.local\fixture-input' --output '.local\fixture-before' --frame-step 1 --pixel-step 2 --min-confidence 1 --max-depth 6 --voxel-size 0.015 --box-margin 0.025 --structure-tolerance 0.025
& $Python scan_pipeline.py edit --scene '.local\fixture-before\scene.json' --object synthetic-target --translate 0.3 0 0.1 --output '.local\fixture-after'
& $Python -m unittest discover -s tests -v
& $Python reconstruct_mesh.py --scene '.local\fixture-before\scene.json' --output '.local\fixture-surface' --voxel-size 0.035 --truncation 0.105
& $Blender --background --factory-startup --python blender_scene.py -- --scene '.local\fixture-surface\surface-scene.json' --output '.local\fixture-blender' --object synthetic-target --translate 0.3 0 0
```

The tests compare the recovered points to the independently authored physical
scene and labels. They also reject a wrong row-major pose, invalid sensor values,
proxy-only motion, and a change to an unrelated object. The fixture's known
ground truth tests implementation correctness; it does not calibrate the real
scanner's physical accuracy.

The published fixture uses resolution scale 4: two 192×128 depth images with
384×256 RGB and 12,284 sampled observations. Its tested full chain produced
9,081 TSDF vertices, 16,820 triangles and three Blender mesh parts. The [published example bundle](../examples/local-scan-fixture/)
contains the original fixture's rendered views and GLB outputs. Those assets
are MIT licensed and can be inspected alongside the generating source.

## Reconstruct and edit the real scan

```powershell
& $Python scan_pipeline.py reconstruct --input $Scan --output '.local\tea-room-observations' --frame-step 1 --pixel-step 2 --min-confidence 1 --max-depth 6 --voxel-size 0.015 --box-margin 0.025 --structure-tolerance 0.025
& $Python reconstruct_mesh.py --scene '.local\tea-room-observations\scene.json' --output '.local\tea-room-surface' --voxel-size 0.035 --truncation 0.105
$Surface = Get-Content -LiteralPath '.local\tea-room-surface\surface-scene.json' -Raw | ConvertFrom-Json
$Target = $Surface.preferred_editable_object
& $Python scan_pipeline.py edit --scene '.local\tea-room-observations\scene.json' --object Sofa0 --translate 0.3 0 0 --output '.local\tea-room-point-edit'
& $Blender --background --factory-startup --python blender_scene.py -- --scene '.local\tea-room-surface\surface-scene.json' --output '.local\tea-room-blender' --object $Target --translate 0.3 0 0
```

Both translations are in meters in the source Y-up world frame. The point
experiment translates Sofa0's selected measurements by 0.3 m on X; the Blender
experiment translates a selected fused surface region by 0.3 m on X. Each
starts from the original scan bundle. The largest eligible surface region is
the Blender entry's default target, and `--object` makes that choice explicit.
Select a different listed UUID to review another region.

The observation entry samples depth pixels at step 2. The TSDF entry uses full
depth images from the selected frames and applies the same confidence and
maximum-depth filters. These are distinct outputs: `--voxel-size 0.015` selects
real points for the observation PLY, while the TSDF voxel size is 0.035 m and its
signed-distance truncation is 0.105 m. RGB is bilinearly resized to depth
resolution; no depth holes are interpolated by the point entry.

Open the saved `before.blend` and `after.blend` in Blender to inspect the object
UUIDs, measured colors, retained capture poses and scene geometry. The script
uses one fixed review camera for `before.png` and `after.png`, saves both Blender
scenes and GLBs, imports both GLBs into fresh scenes, and reopens `after.blend`
for numerical verification. `--no-render` skips PNG generation for a faster
repeat. The capture camera objects preserve poses, not complete calibrated
optical projections; the review render is a fixed visualization view.

## Inspect outputs and repeat offline

| Output | Meaning |
| --- | --- |
| `scene.json` + `observations.npz` | Units, Y-up coordinates, RoomPlan UUIDs/priors, original point colors, frame indices, depth pixel UV, ownership and unresolved candidates |
| `observed.ply`, `objects/*.ply`, `environment.ply`, `ambiguous.ply` | Color point layers; PLY sampling retains an actual measured sample per voxel |
| `reconstruction-result.json`, `edit-result.json` | Numerical inverse projection and saved/reloaded selected-point edit checks |
| `surface-scene.json`, `surface.ply`, per-part PLY | CPU TSDF surface, partition counts, hashes and same-scanner PCD distance comparison |
| `before.blend`, `after.blend`, `before.glb`, `after.glb` | Inspectable original and edited surface scenes |
| `before.png`, `after.png` | Same-view local visual evidence |
| `blender-verification.json` | UUID/role retention, target translation, other world geometry and Blender/GLB round trips |

After preparing all dependencies and input, disconnect using an isolation
method appropriate for your environment and record that method. Repeat the
execution commands into new output directories and compare their result JSONs
with the reported checks. Inverse projection tests numerical consistency;
same-scanner PCD distances test coordinate agreement. Neither is an independent
measurement of object dimensions. TSDF boundary or overlapping-object faces
remain in the environment and are counted in the manifest. Review that count
before interpreting an edit as a complete semantic object move.

The code uses local files during execution and requires no model weights or
service credentials. The measured run's audit guard rejected a deliberately
attempted Python DNS request and blocked Python outbound sockets. It was
installed separately in the reconstruction interpreter and Blender's embedded
Python. Native C networking and independent child processes are outside
that guard; whole-machine network isolation remains unverified. The report
records this scope. Keep scan-derived outputs under ignored
`.local/` directories, and retain original logs and hashes with your local run.

To repeat the recorded process guard, run each Python entry under
[offline_run.py](../offline_run.py), installing the guard separately inside
Blender's Python interpreter:

```powershell
& $Python offline_run.py --script scan_pipeline.py --report '.local\guard-points.json' -- reconstruct --input $Scan --output '.local\guard-observations'
& $Python offline_run.py --script scan_pipeline.py --report '.local\guard-edit.json' -- edit --scene '.local\guard-observations\scene.json' --object Sofa0 --translate 0.3 0 0 --output '.local\guard-point-edit'
& $Python offline_run.py --script reconstruct_mesh.py --report '.local\guard-surface.json' -- --scene '.local\guard-observations\scene.json' --output '.local\guard-surface'
& $Blender --background --factory-startup --python offline_run.py -- --script blender_scene.py --report '.local\guard-blender.json' --blender -- --scene '.local\guard-surface\surface-scene.json' --output '.local\guard-blender' --translate 0.3 0 0 --no-render
```

Each guard JSON should report `dns_block_probe_passed: true`, `status: pass`
and an empty `runtime_denied_network_events` list for the workload. Those
results cover Python's audited operations; perform and document a separate
whole-machine isolation check when that stronger acceptance is required.

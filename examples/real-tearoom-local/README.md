# Real tea-room: local native scene and reproducible evidence

[English measured report](../../reports/REAL-TEAROOM-2026-10-15.md) ·
[中文实测报告](../../reports/REAL-TEAROOM-2026-10-15.zh-CN.md)

The delivery contains 12 editable identity roots and 82 authored structural parts
with the captured cameras and original observed mesh retained. The measured
Table0 correction **fails scene-wide acceptance**; authored structural usability
is checked separately. The generated assets stay local under the source terms.
Use the accepted [synthetic dining scene](../dining-complete/README.md) for an
immediately downloadable native example.

本轮提供真实茶室的本地完整编辑结构场景。观测、来源先验与人工结构补全有不同
标记；真实 Table0 的整体几何修正未通过，编辑场景可用性另行验证。

## Prepare once / 一次准备

Use the existing [software/input preparation instructions](../../docs/REPRODUCING.md)
or [中文准备说明](../../docs/REPRODUCING.zh-CN.md). Obtain the pinned
`LiteReality/example-scans` input under its applicable permissions. The run entry
checks all 253 inventory files against the bundled SHA256 manifest, including
additional or changed frame files. Software/source acquisition can require the
network; it is not part of the local execution command.

Prepare CPython 3.12 with NumPy, Pillow and Open3D from
[requirements-local-lock.txt](../../requirements-local-lock.txt), and Blender
5.1.2. This stage runs on CPU and does not require model weights, CUDA, cloud
compute or API credentials. Existing local environments can be reused.

## Execute / 执行

From the SpatialSceneLab repository root, adapt these executable/input paths to
your machine. The output must be new; repository-local generated assets must
use the ignored `.local/` directory.

在仓库根目录执行。替换解释器、Blender 和已准备输入路径；输出目录须不存在，
不会覆盖上次结果。仓库内生成内容使用被忽略的 `.local/`。

```powershell
$Python = 'C:\path\to\venv312\Scripts\python.exe'
$Blender = 'C:\path\to\blender-5.1.2\blender.exe'
$Scan = (Resolve-Path '.local\example-scans\tea_room').Path
& $Python -B reproduce_real_tearoom.py --input $Scan --output '.local\real-tearoom' --blender $Blender
$ReplayExit = $LASTEXITCODE
Get-Content -LiteralPath '.local\real-tearoom\replay.json'
```

Exit `0`: pipeline completed and measured alignment accepted. Exit **`2`**:
pipeline completed the labelled structural/native delivery, but alignment is
geometrically rejected. Check `completed`, `alignment_gate_passed` and the native
receipt separately. Other exceptions stop the workflow; inspect the retained
stage log. The measured 2026-10-15 delivery exits **2**, as documented.

退出 **2** 是这次真实数据的预期测量结果：本地材料可以完整生成，而整体对齐
没有通过。它不能被当作成功重建。其他执行失败会停止，日志保留在输出目录。

Each subprocess installs the existing Python socket/DNS denial guard. The
alignment guard's expected `SystemExit(2)` describes its failed geometry gate;
the runtime outbound event list must still be empty. OS-level isolation is a
separate verification scope. No stage downloads inputs or installs dependencies.

## Open and inspect / 打开与检查

Open `native/complete-tea-room.blend` in Blender. The scene uses metre units,
retains UUIDs and converts the source Y-up frame to Blender Z-up once. Select
an identity root to move the whole object; select a child to modify an individual
authored part. The authored completion collection and the hidden original
observed-surface collection can be toggled to compare results. Captured cameras
are a separate hidden collection. Inspect per-part provenance/custom properties
before interpreting a proxy as observed real geometry.

在 Blender 打开 `native/complete-tea-room.blend`。根节点编辑整件对象，子节点
编辑结构部件；通过集合切换补全版与原始观测。相机单独保留。保存一份新文件
后可自行修改，原始复现输出继续作为冻结对照。GLB 为
`native/complete-tea-room.glb`。文件中的柜体／椅子是明确标注的结构代理；桌面
有训练观测依据，厚度／桌腿等并非已验证的真实恢复。

## Retained outputs / 产物

| Local output | Purpose |
| --- | --- |
| `source-check.json`, `replay.json`, `*.log`, `*-guard.json` | Inventory, stages, source hashes, runtime/failure evidence |
| `points/`, `surface/` | Original world observations and CPU TSDF surface |
| `baseline/`, `alignment-protocol.json`, `alignment/` | Frozen old control, pre-score protocol, full diagnostic correction and rejection |
| `structural/structural.json`, `structural/public-summary.json` | Authored parameters, provenance, contact/separation/floor and observation mismatch |
| `native/complete-tea-room.blend`, `.glb` | Editable local delivery |
| `native/scene-verification.json`, `native/public-summary.json` | Build/edit/reopen/export evidence |
| `native/tea-room-before.png`, `native/tea-room-complete.png` | Same-view local visual comparison |

Public numerical receipts omit scene geometry, camera poses and transforms.
Reproduction may differ in timings, path-bearing hashes and tiny platform
rounding; compare counts, fixed denominators, gate outcomes, per-frame metrics
and declared tolerances instead of claiming all cross-platform files identical.
Source assets and all derived geometry/images remain local under
[THIRD_PARTY_NOTICES.md](../../THIRD_PARTY_NOTICES.md).

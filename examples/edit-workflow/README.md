# Native editable study and original failure

[English installation / reproduction](../../docs/EDITING-S1.md) ·
[中文使用与复现](../../docs/EDITING-S1.zh-CN.md)

`combined-study-editable.zip` contains authored MIT `before.blend`, `after.blend`,
`after.glb` + its hash-bound identity JSON, the standalone Blender addon and
three actual fixed-camera CPU renders. Its internal and external manifests
record member SHA-256 and sizes. Open `after.blend` with Blender 5.1.2; install
or run the supplied addon to use the AlvenX sidebar. The native file preserves
source-face IDs, owners, manual-review flags and edit journal.

The desk's observed region translated +0.25 m world X / -0.10 m world Y, and
the chair region rotated 32° about world Z around its bounds centre. The two
scripted face assignments validate geometry/appearance preservation. They do
not establish a segmentation improvement; incomplete B0 regions and holes are
visible. The GLB does not carry native per-face edit attributes. Continue
ownership editing from `.blend`, with sidecar lineage in native face order.

`original-verifier-failure.zip` preserves the original S0 combined-study
before/after assets and failed result, plus the original verifier from commit
`321ca429b84595dac1b81590b23dbc934222d38b`. It lets a visitor reproduce the
nearest-source-index false failure and compare the repair on identical bytes.

```powershell
Expand-Archive 'examples\edit-workflow\original-verifier-failure.zip' '.local\original-failure'
$Blender = 'C:\path\to\blender\blender.exe'
& $Blender --background --factory-startup --python-exit-code 1 --python reports/editing-s1/recheck_assets.py -- --project '.local\original-failure' --assets '.local\original-failure' --output '.local\original-check.json'
& $Blender --background --factory-startup --python-exit-code 1 --python reports/editing-s1/recheck_assets.py -- --project '.' --assets '.local\original-failure' --output '.local\repaired-check.json'
```

The first check intentionally records before/after failures; the second passes.
The checker is read-only and reports input hash retention. Geometric triangle
multiset agreement allows seam splitting and vertex reordering; it does not
establish welded seam identity, complete object recovery or physical accuracy.

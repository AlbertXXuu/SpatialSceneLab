# Complete editable dining scene / 完整可编辑餐桌场景

This delivery restores the table and two chairs in the existing public dining
development capture and retains the accepted seven-part stool. The four furniture
objects remain independently selectable. Original observed fragments remain
available in named comparison collections.

本交付使用既有公开餐桌场景的观测，恢复桌子和两把椅子，并保留已验收的七部件
圆凳。四件家具可独立选择、移动和编辑；原始观测碎片保留在命名对照集合中。

This is a synthetic development case with selected furniture families and explicit
structural continuation. Fitting uses captured observations and existing object/
floor priors. Separate original physical geometry is used only for evaluation.

这是合成开发案例，包含明确选择的家具结构和遮挡部分补全。拟合只读取观测和既有
对象／地面先验，原始物理几何仅供独立评估使用。

[Measured English report](../../reports/DINING-COMPLETION-2026-10-13.md) ·
[中文实测报告](../../reports/DINING-COMPLETION-2026-10-13.zh-CN.md) ·
[Shape evaluation](../../reports/dining-completion/evaluation.json) ·
[Layout evaluation](../../reports/dining-completion/layout.json) ·
[Native verification](native/scene-integration.json)

The final source-frame table/west/east validation P95 is 3.692 / 5.668 / 7.570 mm,
with zero original odd-owner points excluded. The final displayed chairs have
separate authored horizontal root translations: west X −0.11187556 m; east
X +0.10492627 m and Z −0.01474643 m in world Y-up metres. Table and stool remain
fixed. Frozen PLYs and original observations retain their source coordinates;
the observation/reference scores describe that source frame, not the translated
display layout. Actual native and GLB geometry passes all 507 cross-furniture
part separation checks at a minimum certified gap of approximately 5.549958 mm.

原坐标下桌子／西椅／东椅验证 P95 为 3.692／5.668／7.570 mm，原归属奇数帧点
排除数为零。最终展示中的椅子另有水平根对象平移：世界 Y-up 米制下，西椅
X −0.11187556 m，东椅 X +0.10492627 m、Z −0.01474643 m；桌子和圆凳固定。
冻结 PLY 和原始观测保留原坐标，观测／参考分数描述原坐标形状，摆放另行验收。
实际 native 和 GLB 的 507 对跨家具部件均通过间隙门槛，最小认证值约 5.549958 mm。

The reference diagnostics also retain a table reverse-area P95 of 48.378 mm,
worse than the original B1's 17.933 mm. Passing the observation and layout gates
does not establish accurate hidden thickness or full physical geometry. See the
reports for the repeated-ray and component-shell area sampling definitions.

参考诊断仍记录桌子反向面积 P95 48.378 mm，比原始 B1 的 17.933 mm 更差。
观测和摆放门槛通过不代表隐藏厚度及全部物理形状准确；重复射线与部件壳面面积
采样的具体口径见双语报告。

## Open and edit / 打开与编辑

Open `native/complete-dining.blend` in Blender. Select a named furniture root in the
Outliner to move, rotate or resize the whole object. Select a child mesh for its
own Transform/Dimensions or enter Edit Mode for mesh changes. Initial fitted
parameters and PLY source hashes are saved as custom properties; these properties
document the imported geometry and do not automatically regenerate it.

在 Blender 中打开 `native/complete-dining.blend`。在大纲视图选中家具根对象，可整体
移动、旋转或缩放；选中子网格可修改其变换、尺寸，或进入编辑模式修改顶点。
自定义属性保存初始拟合参数及来源哈希，用于追踪导入几何，不是自动重新生成
网格的实时参数化 CAD 控制器。

To compare, hide `Completed dining - individually editable components` and show
`Original dining observations + ambiguous fragments (hidden)`. The original stool has its own comparison
collection. Reveal the comparison child objects' Outliner eye toggles as well;
for rendering, enable both the collection and child camera toggles. The
environment is retained. Return to the completed collections
before exporting a scene intended for use.

需要对照时，隐藏完成后的桌椅集合，显示命名的原始桌椅对照集合；圆凳有独立的
原始对照集合，同时开启其中子对象在大纲视图中的眼睛开关；渲染对照时也需开启
集合和子对象的相机开关。环境始终保留。导出使用场景前请切回完成后的家具集合。

## Files and implementation / 文件与实现

- `fitted/`: frozen per-part world-Y-up metre PLY meshes, assembled object meshes,
  fit parameters, evidence selection, input/code hashes and flat-surface control.
- `evaluation-reference/`: evaluation-only physical target triangles, component
  labels and all valid captured rays; it is not a fitting input.
- `evaluation-baseline/`: byte-preserved original B1 target meshes and their
  extraction manifest.
- `native/complete-dining.blend` and `.glb`: editable native scene and visible export.
- `native/dining-before.png`, `native/dining-complete.png`: fixed-camera comparison.
- `native/scene-integration.json`: source retention, independent root editing,
  undo/redo, fresh-process reopen and actual GLB geometry checks.
- `../../reports/dining-completion/`: actual exported-triangle evaluation and
  local execution evidence. Bilingual reports explain the measured results.
- `../../reports/dining-completion/layout.json`: original collision diagnosis,
  authored root translations and a vertex-bound separating-plane proof for every
  cross-furniture component pair.
- `method-controls.zip`: historical failed fit meshes/parameters and captured
  controls. Extract locally and reevaluate with explicit paths; paths and code
  hashes inside these captures remain historical provenance.

`fit_dining.py` selects even-frame candidate observations and fits the selected
structural family. `evaluate_dining.py` separately checks all original odd-frame
owner observations, exported triangles, floor placement and contacts; optional
reference inputs provide further diagnostics. `blender_dining_scene.py` requires
accepted shape and layout evaluations bound to the exact fit/PLY hashes, retains the original
scene, and imports these same triangles into Blender with one proper Y-up to
Z-up rotation. Existing stool and geometry comparison helpers are reused.

`fit_dining.py` 从偶数帧候选观测拟合所选结构；`evaluate_dining.py` 独立检查
全部原始奇数帧对象观测、实际导出三角网格、落地和接触，参考输入提供额外评估。
`blender_dining_scene.py` 只接受与拟合文件及 PLY 哈希绑定的形状与布局通过记录，保留原场景，
经一次正确的 Y-up → Z-up 旋转导入同一组网格。实现复用已有圆凳和几何比较工具。

## Local reproduction / 本地复现

Run from the SpatialSceneLab repository root (`projects/SpatialSceneLab` inside
the AlvenX workspace), with dependencies already installed. The fitting
environment uses `requirements-fit.txt`; the independent evaluator uses
`requirements-local.txt`. Both process offline after setup. Blender 5.1.2 was the
native validation version. Select the respective interpreter on your machine.

在 SpatialSceneLab 仓库根目录（AlvenX 工作区的 `projects/SpatialSceneLab`）
运行；先准备好已有依赖。拟合使用 `requirements-fit.txt`，独立评估
使用 `requirements-local.txt`，准备完成后计算可在本地离线执行。原生验收使用
Blender 5.1.2；按本机情况选择两个环境各自的 Python 解释器。

```powershell
$fitPython = 'C:/ProgramData/anaconda3/python.exe'
$geometryPython = '../../.workspace/cache/spatial-python-312/Scripts/python.exe'
$blenderExe = '../../.workspace/cache/blender/blender-5.1.2-windows-x64/blender.exe'
# Existing Windows conda runtime needs its DLL directory and sequential MKL.
$env:MKL_THREADING_LAYER = 'SEQUENTIAL'
$env:PATH = 'C:/ProgramData/anaconda3/Library/bin;' + $env:PATH

& $fitPython offline_run.py --script fit_dining.py --report .local/dining-fit-guard.json -- --scene examples/stool-fit/input/scene.json --observations examples/stool-fit/input/observations.npz --output .local/dining-fit
& $geometryPython offline_run.py --script evaluate_dining.py --report .local/dining-evaluation-guard.json -- --scene examples/stool-fit/input/scene.json --observations examples/stool-fit/input/observations.npz --fit-dir .local/dining-fit --reference examples/dining-complete/evaluation-reference --baseline-dir examples/dining-complete/evaluation-baseline --output .local/dining-evaluation.json
& $geometryPython offline_run.py --script evaluate_dining_layout.py --report .local/dining-layout-guard.json -- --fit-dir .local/dining-fit --stool-fit-dir examples/stool-fit/fitted --stool-source examples/stool-scene/repaired-dining.blend --propose --output .local/dining-layout.json
& $blenderExe --background --python-exit-code 1 --python offline_run.py -- --blender --script blender_dining_scene.py --report .local/dining-build-guard.json -- --source examples/stool-scene/repaired-dining.blend --fit .local/dining-fit --evaluation .local/dining-evaluation.json --layout .local/dining-layout.json --output .local/dining-native --mode build
& $blenderExe --background --python-exit-code 1 --python offline_run.py -- --blender --script blender_dining_scene.py --report .local/dining-reopen-guard.json -- --source examples/stool-scene/repaired-dining.blend --fit .local/dining-fit --evaluation .local/dining-evaluation.json --layout .local/dining-layout.json --output .local/dining-native --mode reopen
```

Use new output directories; accepted inputs are never overwritten. The Python
network guard denies Python socket/DNS activity and records its probe. It does
not establish OS-level network isolation of native libraries or child processes.
Measured geometry and decisions are reproducible; Blender file bytes may differ
between fresh builds because native serialization is not a deterministic format.

使用新的输出目录，不覆盖已验收输入。Python 网络守卫阻断并记录 Python 的
socket/DNS 操作；它不代表对原生库或子进程实施了操作系统级断网。几何和验收
结论可复现，重新生成的 Blender 文件可能因原生序列化而具有不同字节。

# Editable dining scene / 可编辑餐厅

[Download Blender](repaired-dining.blend) · [Download GLB](repaired-dining.glb) ·
[English report](../../reports/STOOL-TRANSFER-2026-10-09.md) ·
[中文实测报告](../../reports/STOOL-TRANSFER-2026-10-09.zh-CN.md) ·
[Verification JSON](scene-integration.json)

![Complete stool placed in its original dining scene](dining-repaired.png)

## Open and edit / 打开与编辑

Open `repaired-dining.blend` with Blender 5.1.2. On GitHub, binary file pages offer
**Download raw file**. Select `Fitted Stool (structural completion)` to move the
whole stool; expand `Repaired stool - 7 editable structural parts` to select and
edit the seat, legs or braces individually. Save changes under a new filename.
The scene uses metres and Blender Z-up; GLB uses metres and Y-up.

用 Blender 5.1.2 打开 `repaired-dining.blend`。GitHub 二进制页面点击
**Download raw file** 下载。选择 `Fitted Stool (structural completion)`
移动完整凳子；展开 `Repaired stool - 7 editable structural parts`，可逐个
编辑座面、腿和横撑。修改结果另存新文件；改变几何后应重新验收，不能沿用旧分数。

The original observed `Object03` remains hidden in
`Original stool - observed comparison (hidden)`. For a viewport comparison,
hide the repaired collection and reveal `Object03` with its Outliner eye toggle.
For a render comparison, also switch the collection/object camera visibility.
The [before image](dining-before.png) and [after image](dining-repaired.png)
provide a fixed-camera comparison without changing the scene.

原观测 `Object03` 保留在隐藏对照集合中。要在视口比较，先隐藏修复集合，再在
大纲视图中打开 `Object03` 的眼睛开关；要参与渲染，还需切换集合／对象的相机
可见性。上面的修改前后图片使用同一相机，可直接比较。

The seven fitted parts replace only the visible stool. The table, chairs and
environment retain their incomplete observed geometry. Native `.blend` retains
source-face attributes and the hidden comparison; the visible GLB exports 12
meshes and excludes the old stool. Arbitrary per-face Blender attributes are
not part of the GLB guarantee.

这次只替换可见圆凳，桌椅和环境保留原来的不完整观测。`.blend` 保留逐面来源和
隐藏对照；GLB 导出 12 个可见网格，不包含旧圆凳，也不保证任意 Blender 逐面属性。

## Reproduce with existing local tools / 用已有本地工具复现

Run from the repository root. Use new output directories. The measured Windows
machine already had the two Python environments and Blender below; replace these
three executable paths on another machine. Keep the SciPy fitting environment
separate from the pinned Open3D environment (`requirements-fit.txt` versus
`requirements-local-lock.txt`). No model download or API is involved in these
commands. Dependency preparation is separate; full-machine disconnected
execution has not been accepted in this stage.

从仓库根目录运行，使用新的输出目录。以下是实测机器已有的解释器路径；其他机器
只需替换三个可执行文件位置。两个 Python 环境的 NumPy 锁定版本不同，应保持
隔离。命令不调用模型或 API；依赖需要提前准备，本阶段未宣称整机断网验收完成。

```powershell
$FitPython = 'C:\ProgramData\anaconda3\python.exe'
$GeometryPython = 'D:\.Development\AlvenX\.workspace\cache\spatial-python-312\Scripts\python.exe'
$Blender = 'D:\.Development\AlvenX\.workspace\cache\blender\blender-5.1.2-windows-x64\blender.exe'

# Existing Anaconda DLL workaround on the measured host; not an algorithm setting.
$env:MKL_THREADING_LAYER = 'SEQUENTIAL'
$env:PATH = 'C:\ProgramData\anaconda3\Library\bin;' + $env:PATH

& $GeometryPython -B fixtures/make_stool_cases.py --output .local/stool-transfer/cases
& $GeometryPython -B run_stool_transfer.py --suite .local/stool-transfer/cases --output .local/stool-transfer/baseline --fit-python $FitPython --fitter reports/stool-transfer/baseline-fit_stool.py
& $GeometryPython -B run_stool_transfer.py --suite .local/stool-transfer/cases --output .local/stool-transfer/revised --fit-python $FitPython --points-from .local/stool-transfer/baseline
& $GeometryPython -B -m unittest discover -s tests -v
& $FitPython -B -m unittest discover -s tests -p test_stool_fit.py -v
```

The runner retains every case, stdout/stderr, candidate and evaluation. The
revised run reads the baseline point files verbatim. A case return code `2`
can mean a refused fit or an unsuccessful candidate; inspect its reason and
`quality_gate`, not the process code alone. The runner's successful completion
means all attempts/evaluations were recorded, not that every candidate passed.

每个案例的标准输出／错误、候选与评价均保留。修订运行直接读取 baseline 点文件。
案例返回码 `2` 需结合原因和质量门解释；批量命令完成只表示尝试与评价已完成，
不表示所有候选通过。报告保留了旧椭圆误接收及新判定拒绝的对照。

## Rebuild and reopen the dining scene / 重建与重开餐厅

The delivered scene integrates the accepted original stool asset, independently
of the new transfer matrix. These commands use the two bundled inputs without
altering them. The helper extracts only the source `.blend` into a temporary
subdirectory under the requested scratch location and cleans that extraction.
The scratch parent may remain empty; the output retains the usable assets.

餐厅集成使用先前通过验收的原始圆凳，与新迁移矩阵独立。两个输入均已打包，
命令不改写输入。临时源 `.blend` 提取到指定 scratch 下的临时子目录，使用后
自动清理；scratch 父目录可能保留为空，输出目录保留成品与验收记录。

```powershell
$Scratch = Join-Path ([System.IO.Path]::GetTempPath()) 'AlvenX-stool-scene-replay'
& $Blender --background --factory-startup --python-exit-code 1 --python blender_stool_scene.py -- --archive examples/ownership-b1/dining-b1-author-task.zip --fitted examples/stool-fit/native/editable-stool.blend --output .local/stool-scene --scratch $Scratch --mode build
& $Blender --background --factory-startup --python-exit-code 1 --python blender_stool_scene.py -- --archive examples/ownership-b1/dining-b1-author-task.zip --fitted examples/stool-fit/native/editable-stool.blend --output .local/stool-scene --scratch $Scratch --mode reopen
```

Reopen is a new Blender process. It verifies nine source objects, seven fitted
parts, units, native editing/undo/redo, 12 visible GLB meshes, geometry and
appearance. The measured maximum GLB triangle-corner difference is
1.1772695395 × 10⁻⁶ m. This tests file/geometry behavior; it is not a human
usability timing study. To regenerate the standalone stool first, follow the
[original fitting guide](../stool-fit/README.md) and pass your new native asset
as `--fitted`.

重开由新的 Blender 进程执行，检查九个来源对象、七个拟合部件、单位、原生
编辑／撤销／重做、12 个 GLB 可见网格及其几何和外观。实测最大角点差为
1.1772695395 × 10⁻⁶ m。这是文件与几何行为验证，未测量人工使用时长。
如果先重算独立圆凳，按[原拟合说明](../stool-fit/README.md)操作，再将新原生
文件传给 `--fitted`。全部原创合成输入、脚本和资产采用仓库 MIT 许可。

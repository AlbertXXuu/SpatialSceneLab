# Editable stool — download and reproduce / 圆凳下载与复现

[English measured report](../../reports/STOOL-QUALITY-2026-10-09.md) ·
[中文实测报告](../../reports/STOOL-QUALITY-2026-10-09.zh-CN.md)

Open [native/editable-stool.blend](native/editable-stool.blend) in Blender 5.1.2,
or import [native/editable-stool.glb](native/editable-stool.glb) in a glTF viewer.
On GitHub, use **Download raw file** for binary files. The seven named parts are
Seat, Leg 1–3 and Brace 1-2 / 2-3 / 3-1. Select the parent to move the assembly;
select a mesh to edit that part. Saving your edits under a new name preserves
the frozen measured asset. Editing geometry changes the conditions of its checks.

用 Blender 5.1.2 打开 `.blend`，或用 glTF 查看器导入 `.glb`。GitHub 二进制
文件页面点击 **Download raw file** 下载。七个命名部件为座面、三条腿、三根
横撑；选父节点可移动整体，选网格可编辑单个部件。编辑结果另存新文件，以保留
已测量版本。改变几何后应重新核验，不能沿用原始分数。

![Fitted stool](../../reports/stool-quality/fitted-stool-views.png)

## Included files / 内容

- `input/`: exact scene priors and raw sampled observations from the original
  MIT dining fixture. No network access is required to read them.
- `fitted/`: estimated parameters, input hashes and seven individual PLY parts.
- `native/`: editable Blender, GLB and fresh-process verification.
- `evaluation-reference/`: original physical geometry and valid ray hits for
  evaluation only; the fitter does not read this directory.

输入来自原创合成餐厅开发场景；结构族由操作者选择。`input/` 用于拟合，
`evaluation-reference/` 仅用于独立评价。完整资产属于受观测约束的结构拟合，
不是原始传感器已测量的完整表面。当前仅验证一例圆凳。

## Fit locally / 本地拟合

Run from the repository root, with **new output directories**. The optional
fitting environment has NumPy 2.1.3 + SciPy 1.15.3 (`requirements-fit.txt`).
Keep it separate from the Open3D environment whose NumPy version is pinned
differently. Prepare dependencies once; execution uses bundled local files.
For a disconnected machine, transfer a matching Python environment or wheelhouse
first, as described in [the existing guide](../../docs/REPRODUCING.md).

以下 PowerShell 命令从仓库根目录运行。路径变量替换为本机已有 Python／Blender。
可选拟合依赖与 Open3D 环境分开安装，准备依赖后计算只读取本地文件。输出目录
必须是新的或空目录，工具会拒绝覆盖现有结果。

```powershell
$FitPython = 'C:\path\to\fit-env\Scripts\python.exe'
$GeometryPython = 'C:\path\to\open3d-env\Scripts\python.exe'
$Blender = 'C:\path\to\blender.exe'
& $FitPython -B fit_stool.py --scene examples/stool-fit/input/scene.json --observations examples/stool-fit/input/observations.npz --output .local/stool-fit/fitted --self-test
& $FitPython -B -m unittest discover -s tests -p test_stool_fit.py -v
& $Blender --background --factory-startup --python-exit-code 1 --python blender_stool.py -- --fit .local/stool-fit/fitted --output .local/stool-fit/native --mode build --observations examples/stool-fit/input/observations.npz
& $Blender --background --factory-startup --python-exit-code 1 --python blender_stool.py -- --fit .local/stool-fit/fitted --output .local/stool-fit/native --mode reopen --observations examples/stool-fit/input/observations.npz
& $GeometryPython -B evaluate_stool_fit.py --fit .local/stool-fit/fitted/fit.json --reference examples/stool-fit/evaluation-reference --observations examples/stool-fit/input/observations.npz --scene examples/stool-fit/input/scene.json --previous-metrics reports/stool-quality/tsdf-metrics.json --output .local/stool-fit/geometry-check.json --fitter fit_stool.py
```

An existing Anaconda install required `MKL_THREADING_LAYER=SEQUENTIAL` and its
`Library/bin` on that process's PATH; this was a local DLL workaround, not an
algorithm requirement. No dependency installation was necessary for this run.
NumPy/SciPy builds may cause small numeric differences across machines. The
saved JSON records your own hashes and timing; compare geometry and residuals
with the measured report rather than expecting identical timestamps or paths.

本机已有 Anaconda 需要在该进程设置 `MKL_THREADING_LAYER=SEQUENTIAL`，并将
其 `Library/bin` 加入 PATH，以解决已有 DLL 查找问题；这不是算法要求。
不同数值库构建可能产生微小差异；重放记录本机哈希与时间，应比较几何和残差，
而非要求时间戳、绝对路径逐字节一致。

## Repeat TSDF controls / 重建对照

The full RGB-D fixture is generated locally rather than duplicating its images
here. These commands require the existing Open3D environment. The bundled priors
alone cannot reproduce TSDF because their raw-frame paths belong to the measured
host; generate the fresh suite and use its fresh point-scene manifest.

TSDF 对照先生成完整 RGB-D 输入，不能直接依赖打包先验中的历史帧路径：

```powershell
& $GeometryPython fixtures/make_complex_fixture.py --output .local/stool-suite --width 160 --height 120 --scene dev-same-color-dining
& $GeometryPython run_development.py --suite .local/stool-suite --output .local/stool-development
$Scene = '.local/stool-development/dev-same-color-dining/points/scene.json'
& $GeometryPython reconstruct_mesh.py --scene $Scene --output .local/stool-10-fixed --voxel-size .01 --truncation .105
& $GeometryPython reconstruct_mesh.py --scene $Scene --output .local/stool-10-local --voxel-size .01 --truncation .03
& $GeometryPython reconstruct_mesh.py --scene $Scene --output .local/stool-5-local --voxel-size .005 --truncation .015
& $GeometryPython evaluate_stool_quality.py --reference .local/stool-suite/dev-same-color-dining/reference --variant baseline=.local/stool-development/dev-same-color-dining/surface --variant 10-fixed=.local/stool-10-fixed --variant 10-local=.local/stool-10-local --variant 5-local=.local/stool-5-local --output .local/stool-tsdf-check.json
```

## Reproduce the views / 重建展示视角

The renderer uses the same fixed cameras, box crop and neutral material as the
published front/underside images. The front image was replayed and compared
pixel-for-pixel on the measured Blender build, with zero channel difference.
The two-view report figure adds labels to those individual renders.

渲染器复用已发布图像的相机、包围盒裁切与中性材质。公开脚本的正视图在同一
Blender 构建上逐像素复现，通道最大差异为零；报告双视图在单图基础上加了标注。

```powershell
& $Blender --background --factory-startup --python-exit-code 1 --python render_stool_quality.py -- --mesh examples/stool-fit/fitted/fitted_stool.ply --box-prior examples/stool-fit/input/scene.json --object same-stool --output .local/stool-render
```

All data and scripts follow the repository MIT licence. The fixed model family,
development validation, overlapping joints and reference-ray denominator are
documented in both measured reports. They are part of the result's interpretation.

全部数据与脚本采用仓库 MIT 许可；固定结构族、开发验证集、连接处重叠和参考
射线分母的含义均在中英文报告中说明。

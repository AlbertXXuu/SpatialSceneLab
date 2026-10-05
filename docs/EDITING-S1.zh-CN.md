# 复现 Blender 原生 S1 编辑流程

[English](EDITING-S1.md) · [实测报告](../reports/EDITING-S1-2026-10-06.zh-CN.md)

本流程使用本地生成的六组 S0 **B0** 表面分区，检查面归属、刚体编辑、撤销／重做、
原生保存重开与 GLB 导出。按[既有环境说明](REPRODUCING.zh-CN.md)准备 Python 3.12
RGB-D 环境和 Blender 5.1.2。核心执行读取本地文件、使用 CPU；本次新增付费为 0。
Blender 自带其 Python 与 NumPy。交互使用的 [blender_edit.py](../blender_edit.py)
是独立脚本，不需要导入本项目其他模块。

## 打开可编辑样例

下载并解压 [combined-study-editable.zip](../examples/edit-workflow/combined-study-editable.zip)，
检查其中的 README、许可证和文件哈希清单。打开 `before.blend` 从观测分区开始；
打开 `after.blend` 检查已经提交的归属和编辑结果。包内还包含 `after.glb`、对应身份
sidecar，以及用于本次比较的三张实际 CPU 渲染图。

在 Blender 的 Text Editor 中打开包内独立的 `blender_edit.py`（或仓库同名文件），执行 **Run Script**
（`Alt+P`）。在 3D View 用 `N` 显示侧栏，切换到 **AlvenX**。新启动一个 Blender
进程时重新注册脚本。`.blend` 保存观测网格、来源面属性和编辑日志；面板代码单独
提供，没有作为自动执行代码嵌入样例文件。

导入新的准备结果时，先打开空白文档，用 **Import surface bundle** 选择
`surface-scene.json`。导入前检查文件路径、分块 SHA256、唯一区域身份、米制单位、
Y-up 输入及带颜色的三角面。视口中的世界转换为 Blender Z-up。
**Environment + unassigned** 同时包含真实环境几何，以及初始分区未能可靠归入对象的面。

## 检查归属并修改

1. 选择一个导入的来源网格，进入 Edit Mode 和 Face Select，选择非空且未覆盖整个
   来源网格的面子集。在 **Destination** 选择目标，再点击 **Assign selected faces**。
   目标可以是其他观测区域或环境；保持一次只编辑一个来源对象。选中面改变归属，
   同时保留世界位置、逐角颜色、材质和来源面身份。
2. 返回 Object Mode，选择观测对象区域。平移以**米**表示，沿 **Blender 世界 X/Y/Z**；
   旋转以**度**表示，绕**世界 Z 轴**。**Apply metre / degree edit** 使用当前几何
   包围范围中心作为旋转支点，再施加世界平移。这里的平移不是输入的 Y-up 坐标。
3. 用 **Undo**／**Redo** 检查原生状态变化。用 **Save editable .blend** 保存，重开该
   原生文档继续编辑。
4. 用 **Export GLB + identity record** 导出。GLB 使用 glTF Y-up，携带区域身份／来源
   extras；`.identity.json` 把 GLB SHA256 与来源谱系及日志绑定。保留 `.blend`，其原生
   面属性是继续编辑的权威记录。sidecar 面 ID 使用原生面顺序；它不会在 GLB 重导入后
   自动恢复逐面身份。

导出会暂时纳入在对象层隐藏的区域，之后恢复可见性与选择状态。先让 excluded 或受阻
集合在当前 view layer 中可用。捕获到的 sidecar 写入或文件对重命名错误会恢复原有效
文件对；在两次替换之间强制结束进程不构成原子发布。使用前核对 sidecar 资产哈希，并
保留独立保存的原生文档。

## 复现六场景实测任务

在仓库根目录运行 PowerShell。下列可执行文件路径是已经准备好的软件位置占位符，
这些命令不会安装软件。已有 S0 结果时复用校验后的 `surface/` 分块，跳过生成和
重建命令；否则先复现 [S0](DEVELOPMENT-S0.zh-CN.md)：

```powershell
$Python = 'C:\path\to\venv312\Scripts\python.exe'
$Blender = 'C:\path\to\blender-5.1.2\blender.exe'
& $Python fixtures/make_complex_fixture.py --output '.local\s0-inputs' --width 160 --height 120
& $Python run_development.py --suite '.local\s0-inputs' --output '.local\s0-results'
& $Python run_edit_suite.py --results '.local\s0-results' --output '.local\s1-results' --blender $Blender --timeout 1200
& $Python blender_edit_controls.py --blender $Blender --output '.local\s1-controls'
$Result = Get-Content -LiteralPath '.local\s1-results\edit-suite-results.json' -Raw | ConvertFrom-Json
if ($Result.status -ne 'pass') { throw 'Native edit suite failed; retain the run and inspect its logs.' }
```

任务输出须为新目录或空目录，控制实验输出须为新目录。timeout 是每个 Blender 进程
的有限正数秒数。失败保留在结果分母与日志中。只运行一组时给 `run_edit_suite.py`
增加 `--scene dev-combined-study`；默认仅组合书房进行渲染。每组任务由一个后台编辑
进程和一个新启动的原生重开进程组成。输出包括 before／after `.blend`、`after.glb`
及 sidecar、任务／重开 JSON、期望状态数据与日志；组合书房另有 `before.png`、
`after.png`、`after-reimport.png`。

固定任务先把环境的 12 面转给主区域，将主区域平移 `(0.25, -0.1, 0)` m，再把次区域
绕 Z 旋转 32 度，最后把环境的 8 面转给已经变换过的次区域。选面依据面质心到目标
范围中心的距离，不读取参考标签。这些固定选择测量数据保持能力，不测量分割／修正
质量或人工成本。[run_edit_suite.py](../run_edit_suite.py) 的 `TASKS` 固定了每组目标。

内存遥测使用已有的可选 `psutil`，实测版本 7.2.2。缺失时任务仍可执行，内存记录为
不可用。数值是每 0.1 秒采样的 Blender 与已观察子进程的并发 RSS，不是精确峰值或
GPU 显存。不同机器／构建的运行时间与字节哈希可能变化。本套任务没有整机网络隔离
的实测结论。

查阅[公开结果 JSON](../reports/editing-s1/edit-suite-results.json)、
[15 项原生控制](../reports/editing-s1/controls.json)及
[指标 CSV](../reports/editing-s1/metrics.csv)中的分母、源码哈希、进程时间与误差限。
有向三角面比较器在 0.00005 m 角点容差内保留面实例重复次数，几何匹配不能证明焊接
顶点／接缝身份。场景孔洞和仍留在环境里的对象表面仍需归属／重建工作，重新导出
不会修正它们。

## 重建发表图表

可选的 [plot_editing.py](../plot_editing.py) 使用既有绘图环境
（`requirements-plots.txt`），先核对渲染 SHA256，再生成 2400×1080 PNG、PDF、
逐场景 CSV 和图表清单。

```powershell
$PlotPython = 'C:\path\to\plot-environment\python.exe'
& $PlotPython plot_editing.py --results '.local\s1-results\edit-suite-results.json' --assets '.local\s1-results\dev-combined-study' --output '.local\s1-figures'
```

使用新输出目录。本机 Anaconda 绘图环境采用 S0 复现说明中的进程局部
`$env:MKL_THREADING_LAYER = 'SEQUENTIAL'`，与 Blender 核心执行无关。
原始 S0 误报资产也已单独提供下载，可按[原始资产重检说明](../examples/edit-workflow/README.md)
在同一份输入上比较原始失败和修复结果。

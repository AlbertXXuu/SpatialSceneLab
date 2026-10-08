# 将已评测的 B1 分区接入原生编辑

[English](OWNERSHIP-S2A.md)

S2a 将已记录的质心分区接入现有 Blender 工作流，并在同一批固定网格上
解释剩余归属错误。先准备 [S0 CPU 环境与输入](DEVELOPMENT-S0.zh-CN.md)
和 [Blender 5.1.2](EDITING-S1.zh-CN.md)。核心处理使用本地文件与 CPU，
本阶段模型推理和新增付费均为零。

## 导出实际参加评测的分区

在仓库根目录的 PowerShell 中执行，输出使用新目录：

```powershell
$Python = 'C:\path\to\venv312\Scripts\python.exe'
$Blender = 'C:\path\to\blender-5.1.2\blender.exe'
& $Python export_partition.py --results '.local\s0-results' --published-results '.local\s0-results\development-results.json' --output '.local\b1-bundles'
& $Python run_edit_suite.py --results '.local\b1-bundles' --output '.local\b1-editing' --blender $Blender --timeout 1200
```

`--published-results` 指定用来核对准备结果的记录清单。精确重放已发表的 S0
字节时使用 `reports/development-s0-results.json`；新机器重新生成结果时，
使用该次运行自己的 `development-results.json`，如上例。不同 Open3D 构建
可能改变网格顺序；通过新记录的检查不等于与已发表网格逐字节一致。

导出器检查网格、观测清单、原分块和归属数组的哈希，不加载参考标签。每个
来源三角面恰好进入一个输出分块，几何与观测颜色保持。逐分块索引将 Blender
来源面身份关联到原重建网格；另用原生导入检查验证保存文档中的实际映射。

准备 S0 和 B1 后，执行诊断与独立原生来源检查：

```powershell
& $Python partition_diagnostics.py --results '.local\s0-results' --suite '.local\s0-input' --published-results '.local\s0-results\development-results.json' --output '.local\s2a-diagnostics'
& $Python verify_partition_import.py --results '.local\s0-results' --published-results '.local\s0-results\development-results.json' --bundles '.local\b1-bundles' --editing '.local\b1-editing' --output '.local\s2a-lineage' --blender $Blender
```

`--suite` 使用 S0 原输入目录，输出使用新目录。逐面 `face-diagnostics.npz`
包含参考证据，只用于评估，不可用来生成预测归属；数组含义与哈希见
`diagnostics-results.json`。[实测报告](../reports/OWNERSHIP-S2A-2026-10-08.zh-CN.md)
提供原始汇总 JSON/CSV。可选绘图命令使用 Matplotlib 3.10.0：
`python plot_ownership.py --results '.local/s2a-diagnostics/diagnostics-results.json' --output '.local/s2a-figures'`。
仅重新生成图表需要 Matplotlib；固定开发总览要求六场景记录。在本次 Anaconda
环境，绘图 shell 需要 `$env:MKL_THREADING_LAYER='SEQUENTIAL'`。

输出布局为 `<scene>/surface/surface-scene.json`，可交给未修改的 S1 面板和
任务运行器。多盒重叠的面保存为独立的 **Unresolved (-2)** 区域。它沿用已有
区域传输类型，ID、类别和清单状态明确标注未决归属。保存原生文档时同时保留
完整输入包和逐面索引。

## 完成一次作者实际操作

可以直接解压[初始 B1 餐厅任务包](../examples/ownership-b1/dining-b1-author-task.zip)。
内含 `before.blend`、`blender_edit.py`、完整表面包和逐文件哈希。
人工任务只需 Blender 5.1.2；重建和独立来源核验才需要 Open3D。

打开 `dev-same-color-dining` 或 `dev-occluded-bookshelf` 生成的 `before.blend`，
按 [S1 说明](EDITING-S1.zh-CN.md)在 Text Editor 注册 `blender_edit.py`，打开
**AlvenX** 侧栏。从初始 B1 分区开始，不使用脚本已经修改的 `after.blend`。

1. 找到真实目标遗漏。餐厅先检查 **Unresolved (-2)**；书架还需检查环境。
   观察几何再判断归属，诊断参考只提供评估依据，不自动代替修正。
2. 在 Edit Mode 选择来源网格的一部分面，转给目标对象，实时记录选面、
   修正和失误。
3. 回到 Object Mode，用明确的米制偏移移动目标，旋转另一个对象，再通过
   实际界面执行 Undo/Redo。
4. 保存、关闭重开原生文件，导出 GLB 并保留身份 sidecar。检查修正对象、
   剩余未决区域与其他几何。

记录操作者、输入哈希、起止时间、主动修正耗时、选面/归属操作数、撤销次数、
失败恢复和最终文件。有参考标签时在任务结束后评估最终归属。未实际操作的
任务保持 **pending**；脚本耗时、重放次数和自动截图不充当人工成本或易用性结果。

自动 S1 链转移固定的 12/8 个面来验证数据保持，这些选择不代表对象边界已修好。
诊断参考继续保留 S0 的最近表面匹配及法向/可见性限制；没有重建出的表面不计入
分区错误分母。

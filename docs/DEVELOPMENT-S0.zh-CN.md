# 复现六组复杂场景的 S0 开发实验

[English](DEVELOPMENT-S0.md) · [实测报告](../reports/DEVELOPMENT-S0-2026-10-04.zh-CN.md)

从仓库根目录执行。输入是本地生成的原创 MIT 数据，运行此对照无需扫描下载、
账号、GPU 推理或 Blender。按[现有环境说明](REPRODUCING.zh-CN.md)准备 Python
3.12 与离线 wheelhouse。实测核心环境为 Windows、Python 3.12.14、NumPy 2.3.5、
Pillow 12.3.0、Open3D 0.19.0，算法使用 CPU。

```powershell
$Python = 'C:\path\to\venv312\Scripts\python.exe'
& $Python -m unittest discover -s tests -v
& $Python fixtures/make_complex_fixture.py --output '.local\s0-inputs' --width 160 --height 120
& $Python offline_run.py --script run_development.py --report '.local\s0-network-guard.json' -- --suite '.local\s0-inputs' --output '.local\s0-results'
& $Python reevaluate_development.py --results '.local\s0-results' --suite '.local\s0-inputs' --output '.local\s0-sensitivity'
```

每次实验和敏感性复核使用新的或空的输出目录。失败任务保留在 JSON 的尝试分母
中，不能用重跑覆盖来抹去失败。生成器拒绝链接和 junction，保护非空陌生目录；
已有同生成器 suite 可以重新生成。

`suite-manifest.json` 保存场景 ID、开发集身份、随机种子、生成器/配置哈希、
帧数与压力假设。每组 `scan/` 包含测量 RGB-D、标定、带扰动的测量姿态和盒先验；
`reference/` 包含物理几何、独立实例标签、真实姿态与射线命中记录。运行器先完成
三种分区，再加载参考标签，避免算法读取真值。

输出包含固定网格、各方法的面标签/面积指标，以及 `development-results.json`
和 `development-metrics.csv`。场景、表面清单、网格、参考、标签和指标哈希用于
敏感性重放。中间本地清单里的绝对路径记录本次运行，新机器会生成自己的路径；
耗时和系统信息也会变化。不同 Open3D 构建可能改变 TSDF 顶点排序，因此跨机器
比较面积结果并记录构建，而不要求网格文件与本机逐字节相同。

敏感性复核保持网格与算法标签不变，使用 0.04 / 0.08 / 0.12 m 匹配容差及每面
7 / 49 个面积样本，并验证 0.08 m / 7 样本结果与主运行一致。它不会调参或重新
执行分区。不同归属的参考表面近邻距离差不超过 2 mm 时仍记为不可评分。

## 重建图表

绘图属于可选独立环境，核心复现不依赖 Matplotlib。实测绘图环境使用 Python
3.13.5、Matplotlib 3.10.0、NumPy 2.1.3，版本见 `requirements-plots.txt`。

```powershell
$PlotPython = 'C:\path\to\plot-environment\python.exe'
& $PlotPython plot_development.py --results '.local\s0-results\development-results.json' --suite '.local\s0-inputs' --output '.local\s0-figures'
```

本机 Anaconda 的并行 MKL 运行时 DLL 缺失，实际绘图进程使用
`$env:MKL_THREADING_LAYER = 'SEQUENTIAL'` 解决；没有安装库或改变全局配置。
图表清单记录了该值，其他环境无需照搬。场景总览展示独立原创物理参考，柱状图
展示重建表面的分区指标，二者均不能代替可编辑场景产品的任务验收。

## 网络与下一阶段边界

`offline_run.py` 阻断 CPython 出站 socket/DNS，并主动测试阻断生效；它不隔离
原生库、子进程或整台机器。整机断网执行仍需独立验收。[完整评测协议](NEXT-STAGE-PROTOCOL.zh-CN.md)
还要求保留集、可靠的可见性规则、真实房间参考和完整编辑任务。S0 只完成这里
明确限定的六组开发场景对照。

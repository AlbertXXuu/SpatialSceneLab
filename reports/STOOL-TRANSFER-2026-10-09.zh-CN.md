# 圆凳迁移失败分析与可编辑餐厅场景

实测日期：2026-10-09，北京时间 · [English](STOOL-TRANSFER-2026-10-09.md) ·
[下载、编辑与复现](../examples/stool-scene/README.md)

完整圆凳已放回原餐厅，七个部件可以继续编辑，原始观测表面保留为对照层。
六个新增合成开发案例还暴露了一个真实的验收缺陷：旧拟合器接受了形状严重
失真的椭圆座面候选。补充验证误差尾部检查后，这个候选被拒绝，另外三个
适配／压力案例继续通过。本次修改改进的是接受判定，椭圆候选本身的几何没有修好。

![原餐厅中放回完整可编辑圆凳](../examples/stool-scene/dining-repaired.png)

## 预先固定的案例与评价方法

[案例配置](../fixtures/stool-transfer-cases.json)在运行旧拟合器前固定，共六个
种子、每场景一个目标。默认采集为 18 帧、160 × 120 深度分辨率、240° 相机
圆弧、2 mm 高斯深度噪声、1.5% 丢点，无位姿扰动。近似正确的包围盒先验按
物理边界的 1.03/1.02/1.03 倍设置。压力案例改为 75° 视角、6 mm 噪声、8%
丢点；上部裁切案例在 24° 视角下只保留传感器第 0–47 行。这是图像坐标中的
固定裁切，没有用参考实例标签挑选观测。

操作者选择圆形水平座面、三条圆柱斜腿、三根圆柱横撑这一结构族。偶数帧用于
拟合，奇数帧用于验证。运行前独立工程标准为：验证点到实际三角面的 P95 不超过
10 mm、七个独立闭合部件、九处实际实体连接，并核对输入身份和整体／部件一致性。
这些属于开发验证案例，不是未使用过的最终测试集。

拟合结束后，独立评价器才读取单独构造的物理参考，报告 5/10/20 mm 距离阈值，
保留每次尝试。参考召回率的分母是所有有效捕获射线，包含重复观察；它不是完整
表面积或可见表面积覆盖率。反向精度从输出网格按面积均匀采样 100,000 点，固定
种子为 20261009。参考和拟合结构都包含互相重叠的部件壳面及内部面。

## 六个案例的完整结果

下表采用修订运行的实际输出三角面。几何与 baseline 逐字节相同，因此这些几何
数值也适用于旧候选。JSON 分别保留运行时接受判定和独立工程判定。

| 案例 | 预定角色 | 奇数帧观测点 | 三角面 P95，mm | 反向 P95，mm | 反向精度 ≤10 mm | 旧 → 新运行时判定 |
|---|---|---:|---:|---:|---:|---|
| compact-tall | 适配结构，较高较窄 | 1,523 | 3.012 | 2.963 | 98.670% | 接受 → 接受 |
| wide-low | 适配结构，较宽较低 | 2,338 | 3.135 | 2.040 | 98.672% | 接受 → 接受 |
| partial-noisy | 小视角／噪声压力 | 1,689 | 7.182 | 5.714 | 98.892% | 接受 → 接受 |
| four-leg | 不兼容结构族 | 2,088 | 50.401 | 98.370 | 41.726% | 拒绝候选 → 拒绝候选 |
| elliptical-seat | 不兼容结构族 | 1,945 | 12.247 | 392.677 | 49.112% | **接受 → 拒绝候选** |
| upper-crop | 结构证据不足 | 812 | — | — | — | 初始化拒绝 → 初始化拒绝 |

对应的有效参考射线数量依次为 12,603、18,947、13,541、17,075、15,794、6,771。
10 mm 射线召回率在三个接受案例中均为 100%，四腿为 84.340%，椭圆为 93.795%。
上部裁切没有候选，因此对已有射线的召回率为零，距离值不可得，不是零误差。
该案例的 6,771 条参考射线全部落在座面；六个腿／横撑部件的观测数量分别为零，
其逐部件召回率明确记为 null。拟合器实际拒绝原因为：
`Insufficient upper-height separation to initialize the seat band`。

三个接受案例均通过独立七部件闭合和九连接检查。一个小视角带噪案例通过，不能
推广为任意噪声、遮挡或缺失条件都可靠。这组案例很小且由目的明确的配置构成，
不能用来估计实际用户输入的总体失败率。

## 反例与最小修正

椭圆 baseline 通过了原来的均值误差和连接检查，验证点到三角面的平均距离只有
2.274 mm，但 P95 为 12.247 mm，反向面积精度在 10 mm 下只有 49.112%。这说明
较小的平均观测误差可以与严重错误的补全形状同时出现。

运行时唯一新增的判定条件是：奇数帧隐式圆柱残差 P95 ≤10 mm，原有检查仍保留。
椭圆的 12.247482 mm 残差使它被拒绝；独立评价对导出三角面测得 12.247455 mm。
两种距离定义并不普遍等价，尤其是在部件重叠内部。10 mm 三角面目标在运行前
已声明，而将其对应的残差尾部条件加入运行时，是观察失败后的开发修正。

两轮逐例使用完全相同的场景先验和观测字节。五个输出候选共 **40 个 PLY 文件**
的 SHA-256 全部相同；`fit.json` 只改变源码身份、质量门元数据和耗时。上部裁切
两轮均不产生候选。[冻结旧拟合器](stool-transfer/baseline-fit_stool.py)仍可复现。
新增 CLI 回归使用真实椭圆输入，确认旧均值／连接条件满足时，新尾部条件能拒绝它。

第一轮评价遇到过工程错误：初版读取器期待旧格式 `object_statistics`，新生成器
记录的是逐部件射线数量。错误日志和尝试均保留。修正读取器、逐部件核对数量后，
对同一批冻结 baseline 几何产生 `evaluation-v2.json`，没有重新拟合。公开 baseline
评价 JSON 对应这次成功重评。这项评价基础设施修复与拟合器接受判定修正分开记录。
[初次 baseline 运行记录](stool-transfer/baseline-initial-run.json)、
[修订运行记录](stool-transfer/revised-run.json)及[逐例尝试日志](stool-transfer/attempts/)
保留了各进程结果，包括第一次评价失败。

## 放回原餐厅并继续编辑

[餐厅原生文件](../examples/stool-scene/repaired-dining.blend)使用上一阶段已经
通过检查的原始圆凳资产，沿用原有世界坐标追加；这里没有使用新迁移案例的凳子。
原 `Object03` 留在 `Original stool - observed comparison (hidden)` 集合中，
几何仍保留。可见的 `Repaired stool - 7 editable structural parts` 集合包含
拟合父节点与七个子部件。

[集成记录](../examples/stool-scene/scene-integration.json)确认：

- **九个原始场景对象全部保留**。网格、世界变换、来源身份、面归属、颜色和
  材质保持；旧圆凳移入对照集合并隐藏，是有意进行的显示改动。
- 独立 Blender 进程重开通过。脚本将七个拟合部件共同移动 (0.15, 0.10, 0) m，
  原生撤销和重做成功，其他来源几何保持，已交付文件内容不被验收操作改变。
- 可见 GLB 含 **12 个网格**，排除了隐藏的原圆凳。匹配三角面角点的最大误差为
  **1.1772695395 × 10⁻⁶ m**；颜色、材质参数、对象身份和拟合层级均被检查。
- 原生逐面来源信息保存在 `.blend`；GLB 不保证保留任意 Blender 网格属性。
  桌子、椅子和环境仍然是原来的不完整观测几何。

这属于脚本执行的编辑、保存、重开和导出验证，未测量人工使用时长。
[修改前](../examples/stool-scene/dining-before.png)与
[修改后](../examples/stool-scene/dining-repaired.png)使用同一相机渲染。

## 复现与检查

[下载与 PowerShell 操作说明](../examples/stool-scene/README.md)包含生成案例、
旧／新版本同输入运行、以及 Blender 两个独立进程的完整命令。拟合使用已有
NumPy 2.1.3 / SciPy 1.15.3 环境；几何使用 Python 3.12.14 / NumPy 2.3.5 /
Open3D 0.19.0；Blender 为 5.1.2。两个 Python 环境因 NumPy 锁定版本不同而
分开使用。本轮没有新装依赖、GPU 推理或付费服务。

按操作说明定义已有解释器路径后，核心重放命令为：

```powershell
& $GeometryPython -B fixtures/make_stool_cases.py --output .local/stool-transfer/cases
& $GeometryPython -B run_stool_transfer.py --suite .local/stool-transfer/cases --output .local/stool-transfer/baseline --fit-python $FitPython --fitter reports/stool-transfer/baseline-fit_stool.py
& $GeometryPython -B run_stool_transfer.py --suite .local/stool-transfer/cases --output .local/stool-transfer/revised --fit-python $FitPython --points-from .local/stool-transfer/baseline
```

本地全套 **165 项测试中，162 项通过、3 项可选 SciPy 测试在几何环境跳过**；
单独 SciPy 环境的拟合测试 **13/13 通过**，包括实际椭圆 CLI 回归。原生构建和
独立进程重开通过。原始数值见[机器汇总](stool-transfer/summary.json)、
[baseline 逐例评价](stool-transfer/baseline/)及[修订逐例评价](stool-transfer/revised/)。
运行读取已准备的本地文件；这一阶段没有完成整机断网运行验收。

## 方法依据与下一步

[GlobFit（SIGGRAPH 2011）](https://graphics.stanford.edu/~niloy/research/globFit/globFit_sigg11.html)
研究局部 primitive 拟合与全局几何关系的联合处理。
[SPFN（CVPR 2019）](https://openaccess.thecvf.com/content_CVPR_2019/html/Li_Supervised_Fitting_of_Geometric_Primitives_to_3D_Point_Clouds_CVPR_2019_paper.html)
先学习点属性，再估计可变数量的几何 primitive。
[LiteReality-Agent（arXiv v1，2026-10-01）](https://arxiv.org/abs/2610.01863)
提供完整的 agent 室内场景恢复、可编辑场景程序与验证流程参照。这些是方法和
完整能力的依据，不构成当前实现的新颖性证明。本阶段是有限结构族的 CPU 拟合
与场景集成实验，并未完整复现上述算法，也没有神经网络自动结构识别能力。

下一项有价值的工作，是从已有真实扫描选择一个观测足够的对象，检查证据、声明
适用表达，再交付可编辑结果或有理由的拒绝。通用拓扑发现、独立真实形状精度、
人工修正量、全场景补全及产品交互验收仍有待完成。

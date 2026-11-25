# 真实房间家具重建验收与留出深度实测 · 2026-11-25

**整体验收尚未通过。** 目标是从带位姿的真实房间扫描自动重建桌椅，外形、接头和材质接近[展示成品](../examples/dining-showcase/README.md)，并交付可独立编辑的对象。展示成品使用成品网格，验证的是场景交付流程。真实输入的本轮测试覆盖扫描表面在未参与融合的视角上能复现多少实测深度。

## 实验与结果

使用 [LiteReality 茶室的固定采集](../examples/real-tearoom-local/README.md)，保存 62 帧的原始相机位姿。两组使用 31 个偶数帧融合 TSDF，31 个奇数帧共 **1,277,727 条**有效射线用于评估。比较网格首个交点沿相机光轴的深度与实测深度。命中覆盖率和 10／20 mm 内支持率以全部有效射线为分母，误差分位数使用两份网格都命中的同一批射线。物体分组以实测深度点只落入一个原始 RoomPlan 框为准，落入多个框的点仍参加整景统计。 [预先固定的协议](real-tearoom-heldout-2026-11-25-protocol.json) · [完整数值](real-tearoom-heldout-2026-11-25.json)

| 测量 | 默认 35 mm 体素／105 mm 截断 | 候选 20 mm 体素／60 mm 截断 |
| --- | ---: | ---: |
| 整景有效射线命中率 | 93.475% | 93.079% |
| 整景 10 mm 内支持率 | 38.674% | 39.513% |
| 整景 20 mm 内支持率 | 61.596% | 62.430% |
| 整景共同命中绝对误差 P95（1,176,512 条） | 71.538 mm | 81.258 mm |
| Table0 共同命中绝对误差 P95（12,670 条） | 167.271 mm | 202.431 mm |
| 四把椅子共同命中绝对误差 P95（34,196 条） | 215.025 mm | 268.657 mm |
| 网格三角面 | 139,754 | 460,474 |

更细的预设使整景 20 mm 内支持率增加 **0.834 个百分点**，但命中率下降，桌和椅的误差长尾扩大。保留默认预设。候选同时改变体素和截断，本试验衡量的是两套参数组合。RoomPlan 框提供分组线索，采集相机和有效深度来自同一轮扫描。独立物理尺寸、遮挡后方和材质质量仍需要单独证据。 [历史 Table0 几何诊断](TABLE0-ERROR-2026-10-15.zh-CN.md)

## 论文与公开实现

| 路线 | 对本项目的作用和实测约束 |
| --- | --- |
| FIRE3D | 使用带位姿 RGB-D 推理对象和姿态，输出可独立编辑的纹理网格，最接近整体验收目标。发布环境在 Linux、96 GB GPU 上验证，这是作者给出的验证配置；本机 RTX 4060 Laptop 只有约 8 GB 显存，本次未跑原模型。 [论文](https://arxiv.org/abs/2609.08848) · [官方代码](https://github.com/xiahongchi/Fire3D#installation) · [权重](https://huggingface.co/hongchi/Fire3D) |
| LiteReality-Agent | 支持本项目已有的 ScannerApp 数据格式并输出 Blender、GLB。公开流程包括确定性初始化与后续模型驱动的家具生成和修饰。官方本地 GPU 路线在 Linux、至少 24 GB GPU 上测试，完整流程还依赖图像服务和已登录代理。 [论文](https://arxiv.org/abs/2610.01863) · [官方代码及要求](https://github.com/LiteReality/LiteReality-Agent#requirements) |
| ShapeR 与 Scan2CAD | ShapeR 使用已分出的物体、多视图及度量几何做形状补全。Scan2CAD 将库模型对齐到 RGB-D 观测。两者分别提供补全和资产对齐的对照，最终采用哪个候选应由留出视角和独立尺寸决定。 [ShapeR](https://github.com/facebookresearch/ShapeR) · [Scan2CAD 论文](https://arxiv.org/abs/1811.11187) |
| Open3D 与轻量实例掩膜 | 本机已实跑 CPU TSDF；轻量掩膜可作为下一层输入，检测邻桌污染和漏掉的腿、靠背。融合产生的是可见的度量表面，资产或生成补全面再按相同的参考检验。 [Open3D 融合](https://www.open3d.org/docs/release/tutorial/t_reconstruction_system/integration.html) · [SAM 2](https://github.com/facebookresearch/sam2) |

## 复现留出实验

使用 [茶室输入约定](../examples/real-tearoom-local/README.md)和已有 `requirements-local-lock.txt` 环境。下列命令在项目根目录执行，`$scan` 指向有权使用的茶室输入目录。原始及派生几何留在本机。

```powershell
$python = 'C:/path/to/python.exe'
$scan = 'C:/path/to/tea_room'
$run = '.local/real-tearoom-heldout'
& $python -B scan_pipeline.py reconstruct --input $scan --output "$run/points-even" --frame-step 2 --pixel-step 2 --min-confidence 1 --max-depth 6 --voxel-size 0.015 --box-margin 0.025 --structure-tolerance 0.025
& $python -B reconstruct_mesh.py --scene "$run/points-even/scene.json" --output "$run/tsdf-35" --voxel-size 0.035 --truncation 0.105
& $python -B reconstruct_mesh.py --scene "$run/points-even/scene.json" --output "$run/tsdf-20" --voxel-size 0.020 --truncation 0.060
& $python -B evaluate_heldout_surface.py --input $scan --observations "$run/points-even/scene.json" --baseline "$run/tsdf-35" --candidate "$run/tsdf-20" --output "$run/aggregate.json"
```

## 推进条件

先用一组有独立参考的真实桌椅验证，再扩到整个房间。本轮通过 HTTP 范围请求核对了 Apple ARKitScenes 的真实房间 `42897688`。官方标注有两把椅子、一张桌子和一把凳子。**540 个**投影 FARO 深度文件在低分辨率深度和相机内参中均找到同一时间戳，两个抽读时间戳均取得 256 × 192 的设备深度、1920 × 1440 的 FARO 投影深度和内参。物体标注是定向框，目前已核对输入可取得，尚未在该房间执行家具重建。首轮应使用官方投影深度，原始 FARO 点云还需要单独配准。 [Apple 数据格式](https://github.com/apple-aiml-research/ARKitScenes/blob/main/raw/README.md) · [房间公开标注](https://docs-assets.developer.apple.com/ml-research/datasets/arkitscenes/v1/raw/Validation/42897688/42897688_3dod_annotation.json) · [Apple 的原始点云坐标说明](https://github.com/apple-aiml-research/ARKitScenes/issues/41)

先冻结输入视角和评分视角，用未参与拟合的背面、低机位和独立尺寸核对轮廓、桌椅连接和摆放，再检查模型是否保持可编辑层级、贴图和保存重开。候选解释实测后再进入整景比较。当前茶室采用默认融合，保留几何不足和测量范围的记录。

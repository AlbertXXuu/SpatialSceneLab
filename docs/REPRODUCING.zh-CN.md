# 本地 RGB-D 实验复现说明

[English](REPRODUCING.md) · [实测报告](../reports/LOCAL-SCAN-2026-10-04.zh-CN.md)

以下命令从 SpatialSceneLab 仓库根目录在 PowerShell 中执行。当前输入为已准备的
ScannerApp RGB-D 文件夹：`frame_*.jpg`、`depth_*.png`、`conf_*.png`、
`frame_*.json`、参数化 `room.usdz` 和 `pointcloud.pcd`。场景记录实际使用文件的
哈希，表面入口执行前重新验证。解析器接受本文对应的静态 RoomPlan 格式，通用
USD 资产使用 Blender 原生导入。

实测报告记录实际环境和指标。软件、wheel 和输入的首次准备可以联网；之后的
重建、编辑、渲染和验证读取本地文件。

## 一次性准备软件和依赖

准备带 `venv`、`pip` 的 CPython 3.12；实测为 Python 3.12.14，以及 Windows x64
Blender 5.1.2、构建 `ec6e62d40fa9`。软件来源为 [Python 官网](https://www.python.org/downloads/)与
[Blender 官方发行档案](https://download.blender.org/release/)。命令指定可执行文件的
完整路径，因此无需修改 PATH 或共享 Python 环境。

```powershell
$Python312 = 'C:\path\to\Python312\python.exe'
& $Python312 -m venv '.local\venv312'
$Python = (Resolve-Path '.local\venv312\Scripts\python.exe').Path
& $Python -m pip download --only-binary=:all: -r requirements-local-lock.txt --dest '.local\wheelhouse'
& $Python -m pip install --no-index --find-links '.local\wheelhouse' -r requirements-local-lock.txt
& $Python -m pip freeze | Set-Content -Encoding utf8 '.local\installed-packages.txt'
Get-ChildItem -LiteralPath '.local\wheelhouse' -File | Get-FileHash -Algorithm SHA256 | Export-Csv -NoTypeInformation -Encoding utf8 '.local\wheel-sha256.csv'
$Blender = 'C:\path\to\blender\blender.exe'
& $Python --version
& $Blender --version
```

将 Python 安装程序、Blender 压缩包、wheelhouse、wheel 哈希和源码 checkout 一并
保留。可另建 Python 3.12 环境，再执行 `pip install --no-index` 来检查 wheelhouse
能否在不连接包索引的情况下安装。[requirements-local-lock.txt](../requirements-local-lock.txt)
记录实测 Windows x64 / Python 3.12 的 59 包完整依赖，
[requirements-local.txt](../requirements-local.txt)固定三个直接依赖。保留
`installed-packages.txt` 比较本机安装版本；该 Windows lock 不代表其他平台已获验证。

实测 Blender ZIP 的 SHA256 为
`345bedea7b0acf7cc9666423d8553f9129622aea34ded65c23e8cb70f83f14ff`，已对照
官方 Blender 5.1 发行目录中的校验文件。Blender 自带 Python，与重建使用的 venv
分别运行。

## 准备固定版本的真实输入

实测输入为 LiteReality/example-scans 提交
`3f2ca5d618fe86b264716200baea19ae0aea0d49` 下的 `tea_room`。该固定仓库没有
license 文件，请按适用权限从[上游来源](https://github.com/LiteReality/example-scans/tree/3f2ca5d618fe86b264716200baea19ae0aea0d49/tea_room)
自行获取。原始输入及其派生几何、Blender 场景、渲染保留本地，具体范围见
[来源说明](../THIRD_PARTY_NOTICES.md)。

已有 Git 时，可用以下命令只准备选定文件夹：

```powershell
git clone --filter=blob:none --no-checkout https://github.com/LiteReality/example-scans.git '.local\example-scans'
git -C '.local\example-scans' sparse-checkout init --cone
git -C '.local\example-scans' sparse-checkout set tea_room
git -C '.local\example-scans' checkout --detach 3f2ca5d618fe86b264716200baea19ae0aea0d49
$Scan = (Resolve-Path '.local\example-scans\tea_room').Path
```

实测清单为 253 文件、10,518,452 字节，包含 62 组 RGB、深度、置信度与相机记录。
固定 Git checkout 标识输入内容；本地审计还应在运行前记录每个文件的 SHA256。
实测[来源清单](tea-room-source-manifest.json)提供 253 文件的大小与 SHA256。
浅克隆或部分克隆须在断网前取完选定文件，
执行阶段的命令不进行 Git checkout。

按公开 SHA256 清单核对准备好的全部源文件：

```powershell
$InputManifest = Get-Content -LiteralPath 'docs\tea-room-source-manifest.json' -Raw | ConvertFrom-Json
foreach ($ScanEntry in $InputManifest.files) {
    $ScanFile = Join-Path $Scan ($ScanEntry.path -replace '^tea_room/', '')
    $ActualHash = (Get-FileHash -LiteralPath $ScanFile -Algorithm SHA256).Hash.ToLowerInvariant()
    if ((Get-Item -LiteralPath $ScanFile).Length -ne $ScanEntry.bytes -or $ActualHash -ne $ScanEntry.sha256) {
        throw "Input checksum or size mismatch: $($ScanEntry.path)"
    }
}
```

## 运行可公开的原创合成对照

MIT 原创 fixture 从两个相机向两个实体盒、地面和后墙进行物理射线求交，包含
独立语义标签和毫米深度量化。以下步骤无需外部扫描输入：

```powershell
& $Python fixtures/make_rgbd_fixture.py --output '.local\fixture-input' --resolution-scale 4
& $Python scan_pipeline.py reconstruct --input '.local\fixture-input' --output '.local\fixture-before' --frame-step 1 --pixel-step 2 --min-confidence 1 --max-depth 6 --voxel-size 0.015 --box-margin 0.025 --structure-tolerance 0.025
& $Python scan_pipeline.py edit --scene '.local\fixture-before\scene.json' --object synthetic-target --translate 0.3 0 0.1 --output '.local\fixture-after'
& $Python -m unittest discover -s tests -v
& $Python reconstruct_mesh.py --scene '.local\fixture-before\scene.json' --output '.local\fixture-surface' --voxel-size 0.035 --truncation 0.105
& $Blender --background --factory-startup --python blender_scene.py -- --scene '.local\fixture-surface\surface-scene.json' --output '.local\fixture-blender' --object synthetic-target --translate 0.3 0 0
```

测试将反投影点与独立定义的物理场景和语义标签比较，并检出错误的 row-major
相机矩阵、无效传感器值、只移动代理盒的伪编辑，以及其他对象被修改的错误。
合成真值用于检查实现正确性，不校准真实扫描的物理精度。

公开 fixture 使用 `--resolution-scale 4`：两帧 192×128 深度、384×256 RGB，
共 12,284 个采样观测。实测完整链路得到 9,081 个 TSDF 顶点、16,820 个三角面和
3 个 Blender 网格部分。[公开示例包](../examples/local-scan-fixture/)包含原创 fixture 的渲染与
GLB，可以结合生成源码检查；这些原创资产按 MIT 提供。

## 重建并编辑真实扫描

```powershell
& $Python scan_pipeline.py reconstruct --input $Scan --output '.local\tea-room-observations' --frame-step 1 --pixel-step 2 --min-confidence 1 --max-depth 6 --voxel-size 0.015 --box-margin 0.025 --structure-tolerance 0.025
& $Python reconstruct_mesh.py --scene '.local\tea-room-observations\scene.json' --output '.local\tea-room-surface' --voxel-size 0.035 --truncation 0.105
$Surface = Get-Content -LiteralPath '.local\tea-room-surface\surface-scene.json' -Raw | ConvertFrom-Json
$Target = $Surface.preferred_editable_object
& $Python scan_pipeline.py edit --scene '.local\tea-room-observations\scene.json' --object Sofa0 --translate 0.3 0 0 --output '.local\tea-room-point-edit'
& $Blender --background --factory-startup --python blender_scene.py -- --scene '.local\tea-room-surface\surface-scene.json' --output '.local\tea-room-blender' --object $Target --translate 0.3 0 0
```

两处位移均按源 Y-up 世界坐标、以米输入。点实验将 Sofa0 选定的观测点沿 X
移动 0.3 m；Blender 实验将选定融合表面区域沿 X 移动 0.3 m，分别从原始扫描
结果开始。Blender 默认选择三角面最多的可编辑区域，`--object` 将其 UUID 明确
写入命令，也可改为 manifest 内的另一 UUID。

观测点入口每隔两个深度像素采样。TSDF 入口使用选定帧的完整深度图，并应用同样
的置信度、最大深度过滤。两者参数含义不同：观测点的 `--voxel-size 0.015` 每体素
保留一个真实点用于 PLY 展示，TSDF 体素为 0.035 m、符号距离截断为 0.105 m。
RGB 双线性缩小到深度分辨率，点入口不插值补深度空洞。

用 Blender 打开 `before.blend` 和 `after.blend`，查看 UUID、观测颜色、保留的
拍摄相机姿态与场景几何。脚本用同一个固定审查相机生成 `before.png`、
`after.png`，保存两份 Blender 场景和 GLB，将两份 GLB 分别导入空场景，再重开
`after.blend` 做数值回读。加入 `--no-render` 可在复跑时跳过 PNG。
拍摄相机节点保留姿态，不编码完整标定的光学投影；审查图采用固定的展示视角。

## 检查产物与离线复跑

| 产物 | 含义 |
| --- | --- |
| `scene.json` 与 `observations.npz` | 单位、Y-up 坐标、RoomPlan UUID/先验、原始点颜色、帧号、深度像素 UV、归属与未决候选 |
| `observed.ply`、`objects/*.ply`、`environment.ply`、`ambiguous.ply` | 分层彩色点云；每体素保留一个真实采样点 |
| `reconstruction-result.json`、`edit-result.json` | 逆投影数值一致性与保存/重载后的选定观测点编辑检查 |
| `surface-scene.json`、`surface.ply` 与各部分 PLY | CPU TSDF 表面、分区计数、哈希与同一传感器 PCD 的距离对照 |
| `before.blend`、`after.blend`、`before.glb`、`after.glb` | 可检查的编辑前后表面场景 |
| `before.png`、`after.png` | 相同视角的本地视觉证据 |
| `blender-verification.json` | UUID/角色保持、目标位移、其他对象世界几何、Blender/GLB 回读 |

依赖和输入准备完成后，用适合当前环境的隔离方式断开网络并记录方式，再将执行
命令输出到新目录，比较结果 JSON 与报告中的检查。逆投影验证数值一致性，同一
扫描 PCD 距离验证坐标一致性，两者都不提供独立的真实物体尺寸精度。TSDF 边界
面及对象重叠区域保留在环境层，manifest 单独计数；在将编辑理解为完整语义物体
移动前，应检查该计数与本地几何。

代码执行读取本地文件，无需模型权重或服务凭据。实测审计守卫阻断 Python 对外
socket/DNS，并成功拒绝一次故意发起的 DNS 请求；守卫分别安装在重建解释器和
Blender 内嵌 Python 中。原生 C 网络调用与独立子进程不在覆盖范围，整机网络
隔离尚未验证。报告明确该证明范围。
扫描派生资产放在被 Git 忽略的 `.local/` 中，并为本地
实验保留原始日志和哈希。

要复现本次进程级守卫，使用 [offline_run.py](../offline_run.py)运行各 Python
入口，并在 Blender 内嵌 Python 中单独安装守卫：

```powershell
& $Python offline_run.py --script scan_pipeline.py --report '.local\guard-points.json' -- reconstruct --input $Scan --output '.local\guard-observations'
& $Python offline_run.py --script scan_pipeline.py --report '.local\guard-edit.json' -- edit --scene '.local\guard-observations\scene.json' --object Sofa0 --translate 0.3 0 0 --output '.local\guard-point-edit'
& $Python offline_run.py --script reconstruct_mesh.py --report '.local\guard-surface.json' -- --scene '.local\guard-observations\scene.json' --output '.local\guard-surface'
& $Blender --background --factory-startup --python offline_run.py -- --script blender_scene.py --report '.local\guard-blender.json' --blender -- --scene '.local\guard-surface\surface-scene.json' --output '.local\guard-blender' --translate 0.3 0 0 --no-render
```

各守卫 JSON 应有 `dns_block_probe_passed: true`、`status: pass`，工作负载的
`runtime_denied_network_events` 为空。该结果覆盖 Python 可审计的操作；如需整机
断网验收，应另外实际隔离网络、执行并记录结果。

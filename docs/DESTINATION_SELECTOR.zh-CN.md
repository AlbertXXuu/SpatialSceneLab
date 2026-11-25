# Destination 选择器维护

2026-10-25 的修复保留原有归属与刚体编辑流程，使 Destination 标签和实际
`alvenx_id` 一致。旧版三字段动态枚举以当前对象列表索引存储选择：在 Blender
5.1.2 中将首对象 unlink/relink 后，已选 `target-b` 会静默变成 `target-c`；
读取另一 Scene 的菜单也会错误使用全局活动 Scene。10 月 8 日另外记录过 GUI
乱码；源码同时违反 Blender 对动态枚举字符串强引用的要求，但后台测试不能
单独证明历史乱码的唯一成因。

## 实现与旧文件

`blender_edit.py` 继续提供 `Scene.ax_target` EnumProperty。实际选择保存为
Scene 自定义字符串 `ax_target_id`；菜单编号来自冻结 `ax_sources` 的身份顺序，
与对象当前顺序、名称无关。回调使用属性所属 Scene，即使 context 为 None。
模块缓存保留所有返回的 Python tuple/字符串，不保存跨 Undo 失效的 RNA 对象。
get/set 不调用 Blender operator，也不在绘制时改写场景。

选择缺失、身份重复或来源清单无效时显示不可用并禁用归属按钮。执行路径再次
检查目标；整个当前 Scene 的对象身份须为非空字符串且唯一，检查在修改面之前
完成。缺失目标的已存 ID 保留，可在重新链接或 Undo 恢复后继续解析；不选择其他
对象替代它。

旧 `.blend` 只有枚举位置编号时需要重新选择 Destination。编号无法证明原来的
目标身份，因此不猜测迁移。新保存文件及 Undo/Redo 持久化真实 ID。
已有 `scene.ax_target = 'same-table'` 和显式 `target_id` 调用仍可用。
冻结示例中的旧面板、输入、报告保持原字节；维护版单独注册使用。

## 可重建验收

选择器修复及其原生 Blender 回归脚本均随仓库保留。本轮可复验文件版本为：

| 文件 | SHA256 |
| --- | --- |
| `blender_edit.py` | `1b44f52f9a65289fe35529b3572801b4667f0b56fd4cf98c86e742173dcf4e46` |
| `tests/verify_destination_blender.py` | `4871d7298fa14423aaabdb13c151f95274c43e751866723c87b34c308babef1d` |

从仓库根目录在 PowerShell 运行以下命令，将 `$DestinationBlender`
改为 Blender 5.1.2 的可执行文件路径。每个模式由独立 Blender 进程执行。
`$DestinationAudit` 用于留存复跑输出，可指向仓库之外的目录。脚本拒绝
覆盖已有模式输出。

```powershell
$DestinationRepo = (Get-Location).Path
$DestinationBlender = '<Blender 5.1.2 executable path>'
$DestinationAudit = Join-Path $DestinationRepo '.local/destination-selector'
$DestinationVerifier = Join-Path $DestinationRepo 'tests/verify_destination_blender.py'
if (-not (Test-Path -LiteralPath $DestinationBlender -PathType Leaf)) { throw 'Blender executable is missing' }
Set-Location -LiteralPath $DestinationRepo -ErrorAction Stop
$DestinationHead = & git rev-parse HEAD
if ($LASTEXITCODE -ne 0) { throw 'Cannot read the source Git revision' }
$DestinationAddonHash = (Get-FileHash -LiteralPath (Join-Path $DestinationRepo 'blender_edit.py') -Algorithm SHA256 -ErrorAction Stop).Hash.ToLowerInvariant()
$DestinationTestHash = (Get-FileHash -LiteralPath $DestinationVerifier -Algorithm SHA256 -ErrorAction Stop).Hash.ToLowerInvariant()
if ($DestinationAddonHash -ne '1b44f52f9a65289fe35529b3572801b4667f0b56fd4cf98c86e742173dcf4e46' -or $DestinationTestHash -ne '4871d7298fa14423aaabdb13c151f95274c43e751866723c87b34c308babef1d') {
    throw 'Source differs from this reviewed version; review the change before replacing the expected hashes'
}
$DestinationRun = Join-Path $DestinationAudit ('replay-' + (Get-Date -Format 'yyyyMMdd-HHmmss') + '-' + [guid]::NewGuid().ToString('N').Substring(0,8))
New-Item -ItemType Directory -Path $DestinationAudit -Force -ErrorAction Stop | Out-Null
New-Item -ItemType Directory -Path $DestinationRun -ErrorAction Stop | Out-Null
[ordered]@{
    recordedAt = (Get-Date).ToUniversalTime().ToString('o')
    gitHead = $DestinationHead
    addonSHA256 = $DestinationAddonHash
    testSHA256 = $DestinationTestHash
} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $DestinationRun 'source-version.json') -Encoding utf8 -ErrorAction Stop

& $DestinationBlender --background --factory-startup --disable-autoexec --python-exit-code 1 --python $DestinationVerifier -- --mode exercise --output "$DestinationRun/exercise" *> "$DestinationRun/exercise.log"
if ($LASTEXITCODE -ne 0) { throw "exercise failed; inspect $DestinationRun/exercise.log" }

& $DestinationBlender --background --factory-startup --disable-autoexec --python-exit-code 1 --python $DestinationVerifier -- --mode reopen --blend "$DestinationRun/exercise/destination.blend" --output "$DestinationRun/reopen" *> "$DestinationRun/reopen.log"
if ($LASTEXITCODE -ne 0) { throw "reopen failed; inspect $DestinationRun/reopen.log" }

& $DestinationBlender --background --factory-startup --disable-autoexec --python-exit-code 1 --python $DestinationVerifier -- --mode legacy-prepare --output "$DestinationRun/legacy-prepare" *> "$DestinationRun/legacy-prepare.log"
if ($LASTEXITCODE -ne 0) { throw "legacy-prepare failed; inspect $DestinationRun/legacy-prepare.log" }

& $DestinationBlender --background --factory-startup --disable-autoexec --python-exit-code 1 --python $DestinationVerifier -- --mode legacy-reopen --blend "$DestinationRun/legacy-prepare/legacy.blend" --output "$DestinationRun/legacy-reopen" *> "$DestinationRun/legacy-reopen.log"
if ($LASTEXITCODE -ne 0) { throw "legacy-reopen failed; inspect $DestinationRun/legacy-reopen.log" }

foreach ($DestinationMode in @('exercise', 'reopen', 'legacy-prepare', 'legacy-reopen')) {
    $DestinationReport = Get-Content -LiteralPath "$DestinationRun/$DestinationMode/report.json" -Raw -ErrorAction Stop | ConvertFrom-Json
    if ($DestinationReport.status -ne 'pass' -or $DestinationReport.implementation_sha256 -ne $DestinationAddonHash -or $DestinationReport.test_sha256 -ne $DestinationTestHash) {
        throw "Unexpected result or source version in $DestinationMode/report.json"
    }
    $DestinationReport | Select-Object mode, status, passed_checks, blender_version, implementation_sha256, test_sha256
}
Write-Output "Evidence: $DestinationRun"
```

运行顺序与输入关系：

| 模式 | 验收内容 | 留档及下一步输入 |
| --- | --- | --- |
| `exercise` | 默认状态、重复回调/标签缓存、重排、改名、跨 Scene、缺失/恢复、重复身份、注册/注销、原生 Undo/Redo | `exercise/report.json` 与 `exercise/destination.blend` |
| `reopen` | 在新进程读取上一步的新格式存档，验证真实 ID 保持 | 同时读取上一步 `.blend` 及同目录 `report.json`；写入 `reopen/report.json` |
| `legacy-prepare` | 注册旧三字段动态 Enum，经 RNA 明确选择 B 后保存 | `legacy-prepare/report.json` 与 `legacy-prepare/legacy.blend` |
| `legacy-reopen` | 新进程先确认旧 Enum 读回 B，再注册维护版，验证明确重选并写入 Scene 的真实 ID 属性 | 同时读取旧格式 `.blend` 及同目录 `report.json`；写入 `legacy-reopen/report.json` |

两个重开模式都需要准备阶段的报告，不能只拷贝 `.blend`。任一步非零退出即
停止后续命令，检查对应日志；参数、输出目录或导入阶段失败可能尚未生成报告。
结果报告记录实际 Blender 版本、进程、UTC 起止时间和运行时源码/测试哈希。
`source-version.json` 同时保留运行所用 Git HEAD 和文件哈希。

## CI、本机与已有证据的边界

现有 `.github/workflows/ci.yml` 的 `unittest discover` 默认只发现 `test*.py`，
**不会执行 `verify_destination_blender.py`**。CI 对既有 Blender 入口的
`py_compile` 仅做语法检查，也没有运行这四种原生模式。历史 S1/S2a 的单测验证
冻结 ZIP 与原报告的绑定，不能充当维护版选择器的 Blender 运行证据。

因此，修改 `blender_edit.py` 或这份原生验证脚本后，维护验收需另外运行上面
四模式并留档。当前它是本机的显式复现/验收步骤，尚未成为 CI 自动回归门禁。
本次复用已有 Blender，不新增安装 CI 或依赖。synthetic mesh 测试只验证身份
行为；真实菜单可读性、鼠标选择以及用户人工操作成本仍须分别验证。

## 验收结果

四组原生模式各自记录检查数和实际退出状态。各组有重叠。[机器摘要](../reports/destination-selector-2026-10-25.json)

冻结 B1 合成餐厅的历史数据回归分配固定 12 个未决面，来源面总数保持
96,518。原生顶点几何、颜色和材质保持，桌子平移、西椅旋转与撤销重做
通过。独立重开的状态误差为零，GLB 回读的最大世界顶点误差约 1.28 µm。
这些数据评估的是归属操作和原生存档的保持性。

2026-10-25 也在 Blender 5.1.2 实际窗口检查了菜单可读性、选择目标、改名、
撤销重做、GUI 保存与同进程 Revert。18 条状态记录的菜单值、规范 ID 和
解析 ID 一致。软件自动验收仍保留独立进程测试和旧 Enum 回读测试。

## 一手依据

2026-10-25 查阅并固定 Blender v5.1.0 源码；本机执行版本为 5.1.2：

- [动态 Enum 的索引值、借用字符串及警告](https://github.com/blender/blender/blob/v5.1.0/source/blender/python/intern/bpy_props.cc#L2365)。
- [自定义 get/set 与 Blender 5.0 起的存储区别](https://github.com/blender/blender/blob/v5.1.0/doc/python_api/examples/bpy.props.5.py#L4)。
- [Undo 后 RNA 引用失效边界](https://github.com/blender/blender/blob/v5.1.0/doc/python_api/rst/info_gotchas_crashes.rst#L88)。

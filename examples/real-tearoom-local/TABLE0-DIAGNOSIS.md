# Table0 diagnosis replay / 误差诊断复现

This CPU-only stage diagnoses the existing scan and measures a bounded candidate.
It reuses the input and environment documented in [README](README.md). Prepare
`requirements-local.txt` dependencies beforehand; execution performs no downloads,
installs, API/model calls or paid operations. No GPU or Blender is needed here.

本阶段复用既有真实茶室输入与环境，只运行CPU诊断和候选对照。原始输入准备与
第三方条款见[既有说明](README.md)。执行不需要联网服务，不自动安装或下载。

From the repository directory in PowerShell / 在仓库目录执行：

```powershell
$Python = 'C:\path\to\prepared-python.exe'
$Scan = 'D:\LocalData\tea_room'
& $Python -B reproduce_table0_diagnosis.py --input $Scan --output '.local\table0-diagnosis'
```

The output path must be new and separate from the input. Within the repository
it belongs under ignored `.local/`. Keep the complete local output for replay;
raw/derived scan assets are not cleared for public redistribution.

输出目录必须全新且与输入分离；仓库内使用被忽略的`.local/`。保留整份本地结果，
其中扫描衍生点、相机、网格不得随公开源码再分发。

The driver verifies all253 input files, reconstructs original point observations,
audits all639,790 rays and the four fixed input-convention controls, diagnoses the
unchanged training/validation strata, rebuilds the original background baseline,
then freezes and evaluates normal-translation and balanced-pose protocols in
separate guarded invocations. Original/fallback TSDF surfaces must be byte-identical;
the diagnostic candidate must fail before surface output. Logs, per-stage guards,
protocols and `replay.json` stay in the output directory. Exact counts and hashes
are in the [English](../../reports/TABLE0-ERROR-2026-10-15.md) /
[中文](../../reports/TABLE0-ERROR-2026-10-15.zh-CN.md) reports and numeric receipts.

**Expected exit2:** the replay completed, but the geometric candidate was not
accepted. Check `replay.json`: `completed=true`, `alignment_gate_passed=false`,
`failure=null`. An exception or another stage failure is not this expected result.

**预期退出2：** 材料重放完成，但候选没有通过整体几何采用门槛。检查
`completed=true`、`alignment_gate_passed=false`、`failure=null`；其他异常不能算成功。

Output roles / 输出职责：

- `points/`: original reconstructed observations/cameras / 原始点与相机。
- `input-contract/summary.json`, `diagnosis/summary.json`: numeric diagnosis / 数值诊断。
- `normal_translation/`, `balanced_pose/`: frozen controls / 固定对照。
- Inside each control, `scene.json` + `observations.npz`: original identity fallback
  on rejection / 拒绝时保持原始默认输出。
- `candidate-scene.json` + `candidate-observations.npz`: explicit diagnostic layer;
  candidate points and cameras move together / 仅供显式诊断的点和相机层。
- `alignment.json`: detailed local transforms and scores; `summary.json`: numeric
  receipt with transforms/coordinates excluded / 详细本地变换与可公开数值分开。
  Both alignment scene exports are NPZ-only; original PLY layer paths/hashes are
  not advertised there. 原始PLY仍在points/，诊断场景不保留未提供的PLY路径／旧位置哈希。
- `original-surface/`, `fallback-surface/`: verified identical integrated geometry;
  `candidate-surface-must-not-exist/` must be absent / 验证回退几何不变，候选无误输出。

Key implementation files / 关键代码：`audit_table0_inputs.py` validates decoding;
`diagnose_table0.py` preserves strata and denominators; `balanced_table_pose.py`
fits fixed constraints using only even later samples; `audit_real_table_alignment.py`
freezes, scores and exports; `reconstruct_mesh.py` preflights captured pose consistency;
`reproduce_table0_diagnosis.py` orchestrates and checks their actual consumer behavior.
The original native scene remains available through the existing runbook; no
rejected candidate is silently substituted into it.

Focused portable regressions / 回归：

```powershell
& $Python -B -m unittest discover -s tests -p 'test_real_table_alignment.py' -v
& $Python -B -m unittest discover -s tests -p 'test_balanced_table_pose.py' -v
& $Python -B -m unittest discover -s tests -p 'test_rejected_alignment.py' -v
& $Python -B -m unittest discover -s tests -p 'test_table0*.py' -v
```

The optional Open3D matcher/surface cases require the prepared local environment;
the basic numeric/contract cases run in the project's normal portable CI. Python
socket/DNS guard receipts do not certify OS isolation of native libraries.

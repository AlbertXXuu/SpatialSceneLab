# Table0 RGB geometry cross-check / RGB 几何交叉检查

[English measured report](../../reports/TABLE0-RGB-2026-10-25.md) ·
[中文实测报告](../../reports/TABLE0-RGB-2026-10-25.zh-CN.md) ·
[Numeric evidence](../../reports/table0-rgb-2026-10-25.json)

This CPU diagnostic evaluates the **frozen, rejected** October 15 geometric
candidates against original-image correspondences. It fits no camera and never
changes the original scene or the previous adoption gate. The result is
**INCONCLUSIVE**: 385 descriptor-selected matches across 15 fixed pairs, but only
2 pairs meet the declared count/coverage threshold. Correspondence ambiguity
and weak tabletop texture remain material.

本检查用原始 RGB 的对应关系检验冻结候选，不使用深度来建立或筛选对应，也不拟合
新相机。它提供不同于深度拟合目标的图像证据，但原始 ARFrame 相机可能使用同一批
RGB 做视觉跟踪，因此不能当独立物理真值。当前没有足够证据采用候选。

## Prepare once

Use Python 3.12 and an isolated environment. `requirements-rgb.txt` adds optional
OpenCV 4.13.0.92; the core NumPy/Pillow requirements are unchanged. Installation
and obtaining lawful inputs need network if they are not already prepared.
The experiment itself needs no model, service, GPU or credentials.

```powershell
python -m pip install -r requirements-rgb.txt
python -B -m unittest discover -s tests -p test_rgb_pose_check.py -v
```

Obtain the tea-room capture at `LiteReality/example-scans`
`3f2ca5d618fe86b264716200baea19ae0aea0d49` under its source terms, then follow
[the existing diagnosis replay](TABLE0-DIAGNOSIS.md) to produce the original and
both frozen candidate scenes. For example:

```powershell
python reproduce_table0_diagnosis.py --input C:/data/tea_room --output .local/table0-diagnosis
```

That prior replay deliberately exits **2** for the measured rejected geometry
gate. Confirm `replay.json` says `completed: true` and `failure: null`; a crashed
replay is not equivalent. Preserve its `points`, `normal_translation` and
`balanced_pose` outputs. Do not substitute the identity-fallback `scene.json`
for either `candidate-scene.json`.

## Run and inspect

```powershell
python offline_run.py --script rgb_pose_check.py --report .local/rgb-guard.json -- --input C:/data/tea_room --baseline .local/table0-diagnosis/points/scene.json --normal .local/table0-diagnosis/normal_translation/candidate-scene.json --balanced .local/table0-diagnosis/balanced_pose/candidate-scene.json --output .local/table0-rgb
```

Paths may contain spaces when quoted. The output directory must not exist. Run
again with a different output/report path to reproduce; compare `result.json`,
`protocol.json` and the match arrays, excluding `runtime.json` elapsed time.
The published run used Python 3.12.14, NumPy 2.3.5, Pillow 12.3.0 and OpenCV
4.13.0 (wheel 4.13.0.92), one thread, OpenCL disabled. Bitwise equality was checked
in that environment; it is not promised across dependency versions/platforms.

Outputs:

- `protocol.json`: fixed pairs/parameters, source and candidate hashes, code hash
  and versions, written before feature extraction.
- `matches-*.npz`: original-pixel coordinates and descriptor distances, fixed
  before comparing the camera hypotheses. These stay local with the source data.
- `result.json`: all 15 pairs, counts/coverage, error distributions, controls,
  retained errors and diagnostic statuses. `NUMERIC_COVERAGE_ONLY` is not an
  accuracy pass. Degenerate camera scores remain in the denominator as invalid.
- `review-*.png`: deterministic correspondence samples for local visual review;
  never selected by candidate performance and never used to remove score samples.
- `runtime.json` and the separate guard report: measured timing and Python
  socket/DNS guard events. This guard does not isolate native code at OS level.

## Implementation and boundaries

`rgb_pose_check.py` directly reads unrotated 1920×1440 RGB and original K.
RootSIFT mutual bidirectional ratio matches and duplicate-cell suppression use
only image descriptors/coordinates. The old `load_frame()['color']` is unsuitable
here because it resizes RGB to the depth map's 256×192 resolution. Depth files
are checksum-verified for capture identity but their values do not enter matching
or scoring. The fixed candidates were previously fitted using depth.

The evaluator computes `F = inv(Kb).T @ [t]x @ R @ inv(Ka)` and square-root Sampson
error in original-image pixels. R/t come from validated camera rotation/centre
blocks, avoiding tiny non-unit homogeneous-row rounding artifacts. A per-pair
homography is only a descriptive diagnostic; its inliers do not filter scores.
High homography compatibility cannot distinguish a plane from rotation/low
parallax. Low compatibility does not establish a good nonplanar correspondence set.

No new pose, corrected mesh, scene export, millimetre accuracy claim or release
adoption is produced. Original images, correspondence coordinates, camera matrices
and scan-derived review images are not included in the public numeric report.

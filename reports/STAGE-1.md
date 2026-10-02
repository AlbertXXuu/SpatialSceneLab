# Stage 1: object edit/export and reconstruction coordinate integration

Date: 2026-10-02. Verdict: **pass with limits** for this local engineering stage.

The user-authorized first spatial experiment asks whether scene packages can
preserve units, coordinate frames, object identity and edits through export on
the available computer. The smallest acceptance gate is a known three-object
fixture with a real translate/resize operation, a numerical GLB round trip,
falsifying unit/axis controls, and a separate coordinate check using a retained
actual reconstruction. [Machine-readable results](stage-1-results.json) record
the inputs, source hashes, outcomes and limits.

## Actual environment and method

Observed Windows 11 build 26200, Python 3.13.5 and NumPy 2.1.3. `nvidia-smi`
reported an RTX 4060 Laptop with 8188 MiB and driver 596.49. This is GPU inventory;
no CUDA inference was run in this stage. Torch, Depth Anything 3, Open3D, GTSAM,
rosbags, vismatch and trimesh were absent from the current Python environment.
Blender was not found in PATH, targeted Program Files directories, or Windows
uninstall registry entries. No packages, weights or paid services were acquired.

The existing Anaconda native BLAS path faulted with Windows exception
`0xc06d007e` during NumPy matrix multiplication, also under `conda run`.
Matplotlib polygon rendering hit the same native fault. The experiment uses
explicit small-array `einsum` coordinate algebra and the already installed
Pillow for its evidence figures; the shared runtime was not modified.

The controlled fixture has three authored cuboids, explicit meter units, known
dimensions, stable IDs and a right-handed Z-up source frame. Each GLB node
preserves its name, authored origin and TRS-equivalent matrix. The export applies
`(X,Y,Z) -> (X,Z,-Y)` once to produce the glTF Y-up frame in meters. Its JSON
extras preserve source metadata, the exact source-package hash and the rotation.
GLB verification decodes vertex and index buffers and column-major matrices,
then compares physical world vertices against the source scene.

## Observed results

| Check | Result |
| --- | --- |
| Targeted tests | 7 passed, including source hash and non-finite binary corruption rejection |
| Object edit | block-b +0.25 m X; dimensions 0.60×0.90×0.60 → 0.75×0.90×0.60 m |
| Other objects and rotation | preserved |
| Controlled GLB world-vertex round trip | maximum absolute error 0 m, both before and after |
| Equivalent centimeter scene | same physical geometry as meter scene |
| Deliberately wrong unit relabel | 100× mismatch rejected |
| Missing axis conversion | rejected |
| Actual recorded reconstruction | 6,020 sampled points + 4 camera transforms |
| Camera-to-world point equation | max absolute error 5.061982495391248e-7 model m |
| Actual point GLB export/inverse conversion | max absolute error 0 model m |
| Reconstruction integration CPU wall time | 0.1850687 s, one run; no inference |

![Controlled object edit](controlled-edit.png)

The real data is the original GeometryAudit MapAnything Apache / TUM fr1/xyz
four-view window. Four raw NPZ files, the original GLB and selection JSON were
checked against the original SHA256 manifest before reading. A small local audit
exports sampled unsegmented points and camera transform nodes; exact intrinsics,
original camera-to-world matrices and per-view model scale values stay in its
metadata. Camera transform nodes do not encode a complete asymmetric optical
projection. Source NPZ, photographs, full reconstruction and weights stay in
the external original archive; the portable repository carries hashes and
derived metrics only. The original records and retired research candidate remain
separate and unchanged.

The model's world frame is distinct from TUM ground-truth world coordinates.
The explicit `(X,Y,Z) -> (X,-Y,-Z)` rotation matches the original display
conversion; it does not establish gravity alignment. The model-predicted metric
scale is preserved, while object-dimension ground truth is unavailable here.
No segmentation, recovered object edit or new geometry method was demonstrated.

## ScaRF gate and next experiment

ScaRF was **not executed**. Its official offline path requires images plus an
existing trajectory and camera calibration. The current environment lacks its
stack, and native Windows / 8GB support remains unverified. The paper's laptop
was a 12GB RTX PRO 3000 Blackwell; its reported 6-frame performance is not a
measurement on this computer. `is_mono` explicitly allows non-metric trajectories,
so scale consistency is not proof of absolute object dimensions.

Next gate: three independent real scenes, each with calibrated images,
independently measured scale anchors and object dimensions, explicit object
labels/provenance, and original-versus-edited geometry/export comparisons.
Begin with one overlapping six-frame offline sequence only after the CUDA
stack/memory gate is available. Record actual peak GPU memory and wall time then;
compare geometry coverage and dimensions as well as export integrity.

Reproduce the portable checks with the commands in [README](../README.md).
The optional local reconstruction check needs the retained archive described by
GeometryAudit's `REPRODUCING.md`; no fresh model run is implied.

Sources: [ScaRF environment](https://github.com/ori-drs/ScaRF-SLAM/wiki/2.-%F0%9F%93%A6-Environment-Setup),
[offline input](https://github.com/ori-drs/ScaRF-SLAM/wiki/3.-%F0%9F%97%BA%EF%B8%8F-Offline-Reconstruction),
[configuration and outputs](https://github.com/ori-drs/ScaRF-SLAM/wiki/6.-%E2%9A%99%EF%B8%8F-Configuration),
[paper v4](https://arxiv.org/html/2606.00307v4),
[OOM issue](https://github.com/ori-drs/ScaRF-SLAM/issues/2),
[glTF units and transforms](https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html#coordinate-system-and-units).
Dataset/model attribution is in [THIRD_PARTY_NOTICES](../THIRD_PARTY_NOTICES.md).

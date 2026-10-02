<p align="center">
  <img src="docs/assets/alvenx-wordmark.svg" width="320" alt="AlvenX">
  <br>
  <sub>SPATIAL SCENE EXPERIMENTS</sub>
</p>

# SpatialSceneLab

Local experiments for preserving scene coordinates, physical units and object
identity through editing and export. Stage 1 edits three authored rigid cuboids
and verifies their GLB vertex buffers and transforms. A separate local audit
integrates original MapAnything / TUM reconstructed points with their cameras.

Status: **local experiment, 2026-10-02**. Controlled object editing passed;
reconstruction coordinate integration passed. ScaRF inference, recovered-object
segmentation and measured object-dimension accuracy remain untested.

![Controlled edit: three authored objects](reports/controlled-edit.png)

## Run the portable experiment

Use Python 3.11+ with NumPy already available. Tested here on Windows 11,
Python 3.13.5 and NumPy 2.1.3. From this repository:

```powershell
python -m unittest discover -s tests -v
python scene_lab.py export fixtures/three-objects.json examples/three-objects-before.glb
python scene_lab.py edit fixtures/three-objects.json examples/three-objects-edited.json --object block-b --translate 25 0 0 --dimensions 75 90 60 --unit cm
python scene_lab.py export examples/three-objects-edited.json examples/three-objects-after.glb
python scene_lab.py verify examples/three-objects-edited.json examples/three-objects-after.glb
```

The edit translates block-b by 0.25 m on X and changes its X dimension from
0.60 m to 0.75 m. Rotation and the other two objects retain their values.
The JSON declares units and a right-handed Z-up source frame. GLB is meters and
Y-up; the conversion and source scene are preserved in `asset.extras`.
Negative controls detect a 100× unit relabel error and a missing axis conversion.

Inspect [the edited scene package](examples/three-objects-edited.json),
[before GLB](examples/three-objects-before.glb) and [after GLB](examples/three-objects-after.glb).
Pillow is optional for rerendering the evidence figure with `render_evidence.py`.

## Use retained reconstruction evidence

With a local GeometryAudit artifact archive matching its documented layout:

```powershell
python scene_lab.py reconstruction --artifact-root 'C:\path\to\geometry-audit' --output '.local\reconstruction-audit'
python scene_lab.py environment '.local\environment.json'
```

The command verifies original SHA256 values before reading four raw NPZ outputs
and the original GLB. It exports a small sampled point cloud and camera
transforms, then checks their coordinate round trip. Its model-predicted metric
scale remains explicitly unvalidated for object dimensions. No photographs,
source NPZ files, weights or model code are copied into this repository.

[English stage report](reports/STAGE-1.md) · [中文实验报告](reports/STAGE-1.zh-CN.md) ·
[machine-readable results](reports/stage-1-results.json) · [artifact terms](THIRD_PARTY_NOTICES.md)

The next gate is three independently measured real scenes with explicit object
labels, metric references, camera calibration and edit/export comparisons.
ScaRF's [offline workflow](https://github.com/ori-drs/ScaRF-SLAM/wiki/3.-%F0%9F%97%BA%EF%B8%8F-Offline-Reconstruction)
is a candidate after its six-frame CUDA environment and memory gate pass.

Original code and controlled fixture: [MIT](LICENSE).

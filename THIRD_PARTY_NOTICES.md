# Sources and artifact terms

The original CLI, tests, documentation, controlled fixture and its generated
geometry are MIT licensed. External source material retains its own terms.

- NumPy is required for the CLI; Pillow is optional for offline evidence figures.
  Both use their upstream licenses. No dependency code is vendored.
- The copied AlvenX wordmark is the unchanged owner-controlled canonical asset
  from `foundation/brand/assets/alvenx-wordmark.svg`. The MIT grant for code does
  not grant ownership of or an unrestricted trademark license to this mark.
- The optional local reconstruction audit reads the already retained
  **MapAnything Apache** run, using pinned
  [MapAnything code](https://github.com/facebookresearch/map-anything/tree/3d10cf7a3016fc0f9bb13a071ee66c47b10be0d9)
  and [model revision](https://huggingface.co/facebook/map-anything-apache/tree/00f9c245bbcb60522d1ed7f9e9d88462c6e3f38a).
  No upstream code or model weights are copied into this repository.
- **TUM RGB-D Benchmark**: Jürgen Sturm, Nikolas Engelhard, Felix Endres,
  Wolfram Burgard and Daniel Cremers, *A Benchmark for the Evaluation of RGB-D
  SLAM Systems*, IROS 2012. [Official dataset](https://cvg.cit.tum.de/data/datasets/rgbd-dataset),
  [file format](https://cvg.cit.tum.de/data/datasets/rgbd-dataset/file_formats),
  [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
  Changes in the local audit: sample 6,020 model output points from four recorded
  views, preserve predicted camera matrices/intrinsics, and apply the explicit
  display rotation. Original photographs, raw NPZ arrays, and reconstruction GLB
  are not distributed here; only audit metrics and source hashes are included.
- ScaRF-SLAM is cited as a proposed geometry baseline under its
  [GPL-3.0 repository license](https://github.com/ori-drs/ScaRF-SLAM/blob/main/LICENSE).
  This repository contains no ScaRF code or weights and has not executed ScaRF.
- GLB implementation follows the [Khronos glTF 2.0 specification](https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html).

Historical GeometryAudit records are read only and remain separate. Their retired
research candidate is not resumed by this scene packaging experiment.

# Sources and artifact terms

The original CLI, tests, documentation, authored fixtures and geometry generated
from those fixtures are MIT licensed. External source material retains its own
terms; the repository's MIT license does not relicense scan inputs or geometry
derived from them.

- NumPy is required for both local CLIs; Pillow is required for RGB-D image
  decoding and is optional for the historical controlled evidence figure.
  Open3D supplies the optional TSDF integration, and Blender supplies native
  scene saving and glTF I/O. Dependencies retain their upstream licenses and
  are installed separately; no dependency code is vendored.
- The copied AlvenX wordmark is the unchanged owner-controlled canonical asset
  from `foundation/brand/assets/alvenx-wordmark.svg`. The MIT grant for code does
  not grant ownership of or an unrestricted trademark license to this mark.
- The local RGB-D experiment uses the `tea_room` scan from
  [LiteReality/example-scans, commit
  `3f2ca5d618fe86b264716200baea19ae0aea0d49`](https://github.com/LiteReality/example-scans/tree/3f2ca5d618fe86b264716200baea19ae0aea0d49/tea_room).
  Its pinned repository tree contains no license file. Original RGB/depth/
  confidence images, camera records, point clouds, RoomPlan USDZ, and their
  derived PLY, mesh, Blender, GLB and rendered assets remain in local storage.
  The public repository carries source links, hashes, implementation and text
  measurements. Reusers obtain any required input from its source under the
  applicable permissions; this repository supplies no redistribution grant
  for the scan or its derived assets.
- [LiteReality-Agent, commit
  `28ffe1fd4a1b03c93f1877a33a8bc9b806f72fe1`](https://github.com/LiteReality/LiteReality-Agent/tree/28ffe1fd4a1b03c93f1877a33a8bc9b806f72fe1)
  is the reference for the scanner format and scene organization. The local
  experiment implements its own small scan parser and geometric validation;
  it does not vendor the upstream agent, model weights or generated assets.
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
- ScaRF-SLAM is a historical optional geometry candidate under its
  [GPL-3.0 repository license](https://github.com/ori-drs/ScaRF-SLAM/blob/main/LICENSE).
  This repository contains no ScaRF code or weights and has not executed ScaRF.
- GLB implementation follows the [Khronos glTF 2.0 specification](https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html).

The retained GeometryAudit records provide the historical coordinate audit
described in the Stage 1 reports.

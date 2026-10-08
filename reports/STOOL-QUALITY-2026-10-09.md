# From fragmented observations to an editable stool

Measured 2026-10-09 · [中文](STOOL-QUALITY-2026-10-09.zh-CN.md) ·
[Download and reproduce](../examples/stool-fit/README.md)

The previous dining result preserved editable source faces but left the stool
visibly broken. This experiment produces a complete seven-part stool from the
same observations, records the structural assumptions, and tests geometric fit
separately from file/export correctness. It is a single authored synthetic
development case, not an independent real-world reconstruction benchmark.

![Actual fitted assembly, front and underside](stool-quality/fitted-stool-views.png)

## Failure and controlled reconstruction comparison

The MIT `dev-same-color-dining` fixture has 18 RGB-D frames at 160 × 120 pixels,
2 mm Gaussian depth noise, 1.5% pixel dropout and no pose perturbation. The stool
contains a circular seat, three splayed legs and three braces. The original TSDF
used 35 mm voxels and a 105 mm truncation band; leg and brace diameters are 48 mm
and 24 mm. The delivered B1 stool had 13 disconnected fragments. Its ownership
IoU of 0.990428 only scores existing reconstructed faces, so it cannot establish
completeness of missing surfaces.

Three same-input controls were declared before measurement. Their frame-input
hashes match. Reconstruction reads sensor data and priors, not physical reference
geometry. The 10/30 mm setting retains the thin members much better, but all
observed TSDF variants still have leg-to-seat gaps. The 5/15 mm setting introduces
more small holes/splits and does not produce a complete editable assembly.

![Identical-view TSDF controls and evaluation reference](stool-quality/comparison-front.png)

| Voxel / truncation, mm | Reference-ray recall ≤5 mm | ≤10 mm | ≤20 mm | B0 stool area-sampled accuracy ≤5 mm | Reconstruction, CPU s |
|---|---:|---:|---:|---:|---:|
| 35 / 105, original | 72.567% | 87.788% | 96.796% | 56.151% | original run |
| 10 / 105 | 84.883% | 96.151% | 99.928% | 52.610% | 14.70 |
| 10 / 30 | 98.235% | 99.961% | 99.993% | 95.161% | 11.11 |
| 5 / 15 | 99.349% | 99.935% | 99.987% | 97.586% | 36.01 |

Recall uses all **15,354 valid reference ray hits**, including repeated views of
the same regions, and unsigned distance to the full TSDF. It is observation-weighted
coverage, not visible-area or full-surface completeness. Full-surface matches can
include nearby non-stool geometry; the JSON also contains assigned-region scores.
The seat accounts for 12,382 rays (80.643%), legs 2,094 and braces 878. At 5 mm,
leg recall improves from 32.044% to 98.042% and brace recall from 13.667% to 93.508%
for the 10/30 control. Accuracy uses 100,000 seeded uniform-area samples on each
B0 stool region against the physical reference. This is not the B1 task partition.

## Observation-constrained structural fitting

The operator selects the family: a circular horizontal seat, three straight
splayed cylindrical legs and three horizontal cylindrical braces. The fitter
reads only the bundled `scene.json` box/floor priors and `observations.npz` XYZ,
colour, initial ownership and frame indices. Reference geometry is evaluation-only.
No reference vertices or fixture construction parameters are copied into output.

There are 3,812 prior-owned points: 1,927 from even frames train the 26-parameter
model, and 1,885 from odd frames validate it. Bounded SciPy least squares uses a
robust soft-L1 loss, frame/height balancing and three initializations. The selected
run takes 11 function evaluations; no parameter reaches a bound. The final public
CLI replay takes 1.32 s locally. This is one wall-clock observation, not a repeated
performance benchmark. Odd frames participate in acceptance and are **validation**,
not an untouched test set. Family selection itself uses operator knowledge.

The output contains 1,134 vertices and 2,240 triangles in seven independently
editable closed shells. Leg ends extend into the seat and braces into legs to
provide solid contact. These are overlapping parts, not one welded union mesh.
Changes to input points move the estimated stool under the same prior; an
insufficient two-leg example is rejected by the support check. These controls
do not establish reliable rejection of every incompatible furniture family.

## Independent geometric and native-asset checks

The evaluator queries the actual output triangles. These distances supersede
the fitter's implicit-cylinder residuals, which differ in overlapping interiors.

| Measurement | Result |
|---|---:|
| Odd-frame raw observation count | 1,885 |
| Odd-frame point-to-triangle mean / P95 / maximum | 1.047 / 2.954 / 6.294 mm |
| Odd-frame points within 5 / 10 mm | 99.523% / 100% |
| Same reference-ray recall at 5 / 10 / 20 mm | 100% / 100% / 100% |
| Reference-ray mean / P95 / maximum distance | 0.327 / 1.839 / 4.011 mm |
| Seven-reference-part equal-weight mean distance | 0.442 mm |
| Reverse area-sampled accuracy at 5 / 10 / 20 mm | 98.492% / 98.737% / 99.223% |
| Reverse distance P95 / maximum | 1.937 / 25.000 mm |
| Closed parts / boundary edges | 7 / 0 |
| Intended joints with positive solid overlap | 9 / 9 |
| GLB round-trip maximum triangle-vertex error | 0.0 m |

All 1,508 reverse samples beyond 5 mm lie inside the fitted seat, where the fitted
legs extend further inward to make contact. They remain in the denominator.
The reference also consists of overlapping primitive shells, so reverse accuracy
measures proximity to authored shells, not the exposed boundary of a Boolean
solid. Neither a 100% captured-ray score nor contact proves unseen geometry is true.

Contact is independently checked against actual convex face halfspaces: all nine
endpoints have positive interior margins of 23.52–24.53 mm. Actual lowest foot
vertices are 0.000572–0.001173 mm above the floor, within numerical tolerance.
Blender 5.1.2 opens seven named meshes under a parent, with metres and Z-up;
GLB uses metres and Y-up. A fresh process checks native mesh bytes, indices,
parentage, source hashes and oriented triangle geometry after GLB reimport.

## Reproducibility and limits

Fitting used the existing Windows CPU environment with NumPy 2.1.3 and SciPy
1.15.3. TSDF/evaluation used Python 3.12.14, NumPy 2.3.5 and Open3D 0.19.0.
Blender 5.1.2 built and rendered the assets. No model/API purchase or GPU inference
was used. The optional fitting environment is separate from the pinned RGB-D
environment; see the runnable [guide](../examples/stool-fit/README.md).

- [TSDF measurements](stool-quality/tsdf-metrics.json) and [run timings](stool-quality/reconstruction-runs.json).
- [Independent fitted-geometry metrics](stool-quality/fit-metrics.json).
- [Fit parameters, assumptions and input hashes](../examples/stool-fit/fitted/fit.json).
- [Fresh-process native verification](../examples/stool-fit/native/native-verification.json).
- [Figure provenance and camera/crop settings](stool-quality/render-record.json).
- [Fitter](../fit_stool.py), [geometry evaluator](../evaluate_stool_fit.py),
  [native builder](../blender_stool.py) and [failure controls](../tests/test_stool_fit.py).

Source code, bundled inputs, output assets and evaluation reference are released
under the repository MIT licence. Absolute paths in measured JSON identify the
original local run; replay tools use explicit supplied inputs and content hashes.
Local regression discovery ran 146 tests: 144 passed and the two SciPy fitting
controls were skipped in the Open3D environment. All 12 fitting tests then passed
in the separate SciPy environment, including those two controls. The public fit
also replayed successfully under the Python outbound socket/DNS guard. This
guard is narrower than whole-machine network isolation; whole-machine disconnected
acceptance is pending. The replayable front render matched the measured front
image pixel-for-pixel on this Blender build.

This result solves the visibly broken stool in one bounded development example.
It does not complete the table/chairs, infer arbitrary object topology, establish
real-scan generalization, measure human editing time or provide a physics-ready
welded solid. The next useful validation is a changed stool shape and an
incompatible-family case, followed by a real observed object. Historical S0/S1/S2a
results remain frozen and are not replaced by these scores.

# Stool transfer failures and an editable dining scene

Measured 2026-10-09, Beijing time · [中文](STOOL-TRANSFER-2026-10-09.zh-CN.md) ·
[Open, edit and reproduce](../examples/stool-scene/README.md)

The complete stool now sits in its original dining scene, with seven editable
parts and the original observed surface retained as a comparison layer. Six new
synthetic development cases also expose a real acceptance defect: the original
fitter accepted an elliptical seat despite a badly distorted completion. Adding
a validation-tail check rejects that candidate while preserving the three
accepted compatible/stress cases. The change improves the decision; it does not
repair the rejected ellipse's geometry.

![Original dining scene with the completed editable stool](../examples/stool-scene/dining-repaired.png)

## Frozen cases and measurement protocol

The [case configuration](../fixtures/stool-transfer-cases.json) was fixed before
running the existing fitter. It has six seeds and one target per scene. Default
capture is 18 RGB-D frames at 160 × 120 pixels, a 240° camera arc, 2 mm Gaussian
depth noise, 1.5% dropout and no pose perturbation. Near-correct box priors are
1.03/1.02/1.03 times the physical bounds. The stress case uses a 75° arc, 6 mm
noise and 8% dropout. The upper-crop case keeps sensor rows 0–47 on a 24° arc;
this is a sensor-space mask, not a mask selected from reference instance labels.

The operator selects one family: circular horizontal seat, three cylindrical
splayed legs and three cylindrical braces. Even-frame observations determine
the fit; odd frames validate it. The predeclared independent engineering target
is validation point-to-triangle P95 ≤10 mm, seven individually closed parts and
nine actual solid contacts. Input identities and assembly consistency are also
checked. These are development validation cases, not an untouched test set.

The evaluator reads independently authored physical geometry only after fitting.
It reports nearest-triangle distances at 5/10/20 mm, preserving every attempt.
Reference recall counts all valid captured rays, including repeated views;
it is not full-surface or visible-area completeness. Reverse accuracy uses
100,000 uniform-area output-mesh samples, seed 20261009. Both reference and
fitted models contain overlapping primitive shells, including internal faces.

## All six cases

The table uses actual output triangles from the revised run. Geometry is
byte-identical to the baseline, so these measurements also describe the baseline
candidates. Runtime acceptance and independent engineering acceptance remain
separate in the JSON.

| Case | Predeclared role | Odd-frame points | Triangle P95, mm | Reverse P95, mm | Reverse accuracy ≤10 mm | Baseline → revised runtime decision |
|---|---|---:|---:|---:|---:|---|
| compact-tall | Compatible changed dimensions | 1,523 | 3.012 | 2.963 | 98.670% | accept → accept |
| wide-low | Compatible changed dimensions | 2,338 | 3.135 | 2.040 | 98.672% | accept → accept |
| partial-noisy | Limited-view/noise stress | 1,689 | 7.182 | 5.714 | 98.892% | accept → accept |
| four-leg | Incompatible family | 2,088 | 50.401 | 98.370 | 41.726% | reject candidate → reject candidate |
| elliptical-seat | Incompatible family | 1,945 | 12.247 | 392.677 | 49.112% | **accept → reject candidate** |
| upper-crop | Insufficient structural evidence | 812 | — | — | — | refuse initialization → refuse initialization |

Valid reference-ray counts are respectively 12,603; 18,947; 13,541; 17,075;
15,794; and 6,771. At 10 mm, recall is 100% for each accepted case, 84.340%
for four-leg and 93.795% for elliptical-seat. Upper-crop has no candidate, so its
recall for existing rays is zero; its distances are unavailable, not zero error.
All 6,771 upper-crop rays hit the seat. Each of the six leg/brace primitives has
zero observed rays and an explicit null per-part recall. The fitter's actual
refusal is `Insufficient upper-height separation to initialize the seat band`.

The three accepted cases pass independent seven-part closure and nine-contact
checks. Acceptance of one noisy limited-view case does not establish performance
for arbitrary noise, occlusion or missing parts. The sample is too small and
purposefully constructed to estimate a population failure rate.

## Counterexample and minimal correction

The ellipse baseline passed its mean-error and contact checks. Its validation
point-to-triangle mean was only 2.274 mm, while P95 was 12.247 mm and reverse
area accuracy at 10 mm was 49.112%. This demonstrates why a small average
observation error can coexist with a wrong completed shape.

The sole runtime decision change requires odd-frame implicit-cylinder residual
P95 ≤10 mm, alongside the existing checks. Its 12.247482 mm residual rejects
the ellipse. The independent evaluator separately measures 12.247455 mm against
the exported triangles; the two distance definitions are not generally identical,
especially within overlapping solids. The triangle target was declared before
the runs; adding the runtime proxy check is a development response to the failure.

Both runs use the same per-case scene and observation bytes. All **40 PLY files**
across the five generated candidates have identical SHA-256 hashes. Only source
identity, quality-gate metadata and timing change in `fit.json`. The upper-crop
case produces no candidate in either run. The
[frozen baseline fitter](stool-transfer/baseline-fit_stool.py) remains available.
The CLI regression replays the actual elliptical input and confirms rejection
despite passing the old mean/contact conditions.

The first evaluation attempts failed because the initial loader expected the
older `object_statistics` field, while the new generator records per-primitive
ray counts. Those execution errors and attempts were retained. The loader was
corrected to validate each primitive count; the same frozen baseline geometry
was re-evaluated as `evaluation-v2.json`, without refitting. Public baseline
evaluation JSONs contain that successful re-evaluation. This infrastructure
repair is separate from the fitter acceptance correction.
The [initial baseline run record](stool-transfer/baseline-initial-run.json),
[revised run record](stool-transfer/revised-run.json) and
[per-case attempt logs](stool-transfer/attempts/) preserve process outcomes,
including the initial evaluation failures.

## Integration into the original scene

The [native dining file](../examples/stool-scene/repaired-dining.blend) appends the
previously accepted original stool in its existing world frame. It does not use
one of the new transfer candidates. The source `Object03` stays in
`Original stool - observed comparison (hidden)`; its geometry is retained.
The visible `Repaired stool - 7 editable structural parts` collection contains
the fitted parent and its seven children.

The [integration record](../examples/stool-scene/scene-integration.json) verifies:

- All **nine original scene objects** are retained. Meshes, world transforms,
  source identities, face assignments, colours and materials are preserved;
  the old stool's comparison collection and visibility are the intended changes.
- Fresh-process Blender reopening verifies the saved scene. A scripted native
  edit moves all seven fitted parts by (0.15, 0.10, 0) m; undo and redo succeed,
  unrelated source geometry stays unchanged, and the delivered file stays frozen.
- The visible GLB contains **12 meshes**, excluding the hidden original stool.
  Its maximum matched triangle-corner error is **1.1772695395 × 10⁻⁶ m**.
  Colours, material values, object identities and fitted hierarchy are checked.
- Native per-face lineage remains in `.blend`; arbitrary Blender mesh attributes
  are not guaranteed by GLB. The table, chairs and environment retain their
  original incomplete observed geometry.

This is a scripted edit/save/reopen/export verification, not a timed human
usability study. Compare the [before](../examples/stool-scene/dining-before.png)
and [after](../examples/stool-scene/dining-repaired.png) renders from one camera.

## Reproduction and verification

The [download and PowerShell guide](../examples/stool-scene/README.md) includes
the complete commands for generation, baseline/revised replay and two separate
Blender processes. Fitting used the existing NumPy 2.1.3 / SciPy 1.15.3 environment;
geometry used Python 3.12.14 / NumPy 2.3.5 / Open3D 0.19.0. Blender is 5.1.2.
These two Python environments remain separate because their NumPy pins differ.
No new dependency installation, GPU inference or paid service was used.

After defining the existing interpreter paths as in the guide, the central replay is:

```powershell
& $GeometryPython -B fixtures/make_stool_cases.py --output .local/stool-transfer/cases
& $GeometryPython -B run_stool_transfer.py --suite .local/stool-transfer/cases --output .local/stool-transfer/baseline --fit-python $FitPython --fitter reports/stool-transfer/baseline-fit_stool.py
& $GeometryPython -B run_stool_transfer.py --suite .local/stool-transfer/cases --output .local/stool-transfer/revised --fit-python $FitPython --points-from .local/stool-transfer/baseline
```

The local full suite ran **165 tests: 162 passed and three optional SciPy tests
were skipped** in the geometry environment. The separate SciPy fitting suite
passed **13/13**, including the real ellipse CLI regression. Native build and
fresh-process reopening passed. See [machine summary](stool-transfer/summary.json),
[baseline evaluations](stool-transfer/baseline/) and
[revised evaluations](stool-transfer/revised/).
Execution uses prepared local files; a whole-machine disconnected run has not
been accepted or measured in this stage.

## Method context and remaining work

[GlobFit (SIGGRAPH 2011)](https://graphics.stanford.edu/~niloy/research/globFit/globFit_sigg11.html)
studies coupling primitive fitting with global geometric relations.
[SPFN (CVPR 2019)](https://openaccess.thecvf.com/content_CVPR_2019/html/Li_Supervised_Fitting_of_Geometric_Primitives_to_3D_Point_Clouds_CVPR_2019_paper.html)
learns point properties and estimates a variable set of primitives.
[LiteReality-Agent (arXiv v1, 2026-10-01)](https://arxiv.org/abs/2610.01863)
is a complete agentic indoor-scene reconstruction reference with editable scene
programs and verification. These works establish useful methods and system
context; they do not establish novelty for this implementation. This stage is a
bounded CPU fitting/integration study, not a full reproduction of any of those
systems or a neural structure-discovery method.

The next useful step is one sufficiently observed object from the existing real
scan: inspect its evidence, choose and state an appropriate representation, and
measure a usable editable result or a justified refusal. General topology
discovery, independent real-world shape accuracy, manual correction effort,
full-scene completion and product-level usability remain open.

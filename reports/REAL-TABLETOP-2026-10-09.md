# A real tabletop: ownership ambiguity and cross-view inconsistency

Measured 2026-10-09, Asia/Shanghai · [中文](REAL-TABLETOP-2026-10-09.zh-CN.md)

The first real-object follow-up identifies why a usable tabletop cannot yet be
accepted. A fixed training-view plane improves substantially over the RoomPlan
box top, but does not agree with later views. Background-only registration
improves the mean while worsening the error tail. The original acceptance target
is unmet; no accepted real tabletop or complete table asset is delivered here.

## Input and observed problem

Use the already prepared 62-frame `tea_room` RGB-D scan and the frozen point
reconstruction from the [local scan experiment](LOCAL-SCAN-2026-10-04.md).
Table0 is `46C25048-47D3-45BE-BEDF-D3DF0DF758F6`. Of its 15,361 box-candidate
observations, 9,912 (64.53%) also fall into chair priors: 2,083 with Chair1 and
7,829 with Chair3. Only 5,449 observations have unique Table0 ownership. Overlap
ambiguity is a property of the supplied boxes, not a measured segmentation error.
Repeated views are counted as observations, not independent physical points.

The existing table surface has 708 vertices, 895 triangles and 43 connected
components. Its largest components contain 439, 142, 76 and 64 triangles. This
supports investigating observed tabletop recovery before attempting structural
completion. Lower observations do not establish four complete table legs: two
local corner groups have support only from frames 56–58, and may include chair
or floor points. RGB inspection also confirms that this room does not supply a
matching instance of the previously selected three-leg circular-stool family.

## Fixed experiment

The input, candidate membership, frame split and validation denominator are fixed
before registration. Train on frame indices ≤30 and validate on indices >30.
Use Table0 candidate observations in the prior-local height band from 150 mm
below to 50 mm above the box top. This gives 5,262 training and **7,127 validation
observations**. Neither set is an independent semantic annotation.

Fit a near-horizontal plane `y = a*x + b*z + c` in the object prior's coordinate
system using 1,000 seeded triplets, 10 mm residual support, frame-balanced
weights and five least-squares refinements. Triplet planes above 10° tilt are
excluded; least-squares refinements are unconstrained.
All validation-band observations, including outliers, remain in reported errors.
The bounded support target is at least 80% within 10 mm across at least five
validation frames; this is a declared engineering diagnostic, not a calibrated
physical-accuracy guarantee. The fitted plane's observed tilt is about 0.247°.

An independent local-grid diagnostic compares height medians in shared 5 cm XZ
cells with at least three observations per frame. Relative to frame 4, frames
46/47/48/49 have 78/95/66/30 common cells, respectively. Every common cell has a
positive height difference; the median differences are about 17.12/13.11/17.54/
14.32 mm. Unique-owned and ambiguous observations show similar offsets. This
supports cross-view geometric inconsistency beyond a simple ownership-mixing
explanation; it does not distinguish pose error, depth bias or physical motion.

## A background-only registration control

Freeze the trained plane and original validation indices. Exclude every original
object OBB, expanded by 80 mm on each local axis, from both registration source
and target. The target contains only training-view background. Each later frame
is independently aligned with point-to-plane ICP: 30 mm voxels, 120 mm target
normal radius/30 neighbours, 30 mm Tukey loss, 60 mm correspondence limit and 60
iterations. Table observations never enter registration.

A transform is applied only when background fitness is at least 0.5, there are
at least 150 correspondences, the normal-Jacobian eigenvalue ratio is at least
1e-4, translation is at most 80 mm, rotation at most 3°, and matched RMSE does
not increase. Unaccepted transforms remain identity; their table observations
stay in the denominator. These thresholds are controls for this experiment, not
a universal registration-validity test. Estimated poses never overwrite inputs.

| Geometry compared with the same 7,127 observations | Mean distance | P95 distance | Within 10 mm |
|---|---:|---:|---:|
| Original RoomPlan box-top plane | 68.160 mm | 91.359 mm | 0.617% |
| Fixed plane fitted from training observations | 15.489 mm | 26.409 mm | 22.759% |
| Same plane after background-only ICP | 11.787 mm | **35.510 mm** | 58.117% |

The mean decreases by about 24% after ICP, but P95 increases by about 34%. The
80% support target still fails. Frame 47 improves from 25.2% to 92.3% within
10 mm, whereas frame 46 worsens from 15.8% to 4.6%. A better background fit does
not guarantee a better target-object fit. We retain all outcomes rather than
choose transformations using table validation scores.

## Reproduce

The [audit CLI](../audit_real_tabletop.py) reads the two prepared files and writes
a new output directory. Use the existing Open3D environment from
[the source preparation guide](../docs/REPRODUCING.md); no model or API is called.
From the repository root in PowerShell:

```powershell
$Python = 'D:\.Development\AlvenX\.workspace\cache\spatial-python-312\Scripts\python.exe'
$Observed = 'D:\.Development\AlvenX\.workspace\experiments\spatial-local-20261004\points-before'
& $Python -B audit_real_tabletop.py --scene "$Observed/scene.json" --observations "$Observed/observations.npz" --object 46C25048-47D3-45BE-BEDF-D3DF0DF758F6 --output .local/real-tabletop
& $Python -B -m unittest discover -s tests -p test_real_tabletop.py -v
```

Replace the executable/input paths on another machine. Use a fresh output
directory. `plane-fit.json` and `background-icp.json` retain detailed local
diagnostics; `comparison.json` reports the fixed denominators, input/source
hashes, controls and per-frame outcomes. Its successful creation does not mean
the geometry passes. [Published measurements](real-tabletop/comparison.json)
contain the numerical comparison. Numerical differences from platform/ICP
rounding should be interpreted separately from changes in inputs or parameters.

The input scan's pinned source has no redistribution license. Raw RGB/depth,
camera records and derived geometry remain local under
[the existing source policy](../THIRD_PARTY_NOTICES.md). The public delivery
contains original code and numerical/text evidence. The prior synthetic
[editable dining scene](../examples/stool-scene/README.md) remains the accepted
downloadable asset; it is not presented as a reconstruction of this real table.

## What this changes next

Continue with the same real object, first addressing agreement between views and
the table/chair boundary. Compare a target-aware local alignment with the frozen
background-only control, and examine both the table and neighbouring geometry
before completion. Existing results now serve as development evidence; a future
independent capture is needed for a final generalization test. Ground-truth
dimensions, semantic masks, unseen surfaces, manual correction time and full
product interaction have not been measured in this stage.

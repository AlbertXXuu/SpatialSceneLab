# Reproduce the native Blender S1 edit workflow

[中文](EDITING-S1.zh-CN.md) · [Measured report](../reports/EDITING-S1-2026-10-06.md)

This workflow starts from the six locally generated S0 **B0** surface bundles.
It tests face ownership, rigid edits, undo/redo, native save/reopen and GLB export.
Prepare the existing Python 3.12 RGB-D environment and Blender 5.1.2 using
[the environment guide](REPRODUCING.md). Core execution uses local files and CPU;
the measured run allocated no new paid resources. Blender supplies its own Python
and NumPy. Interactive use of [blender_edit.py](../blender_edit.py) is standalone
and requires no imports from the rest of this project.

## Open an editable example

Download [combined-study-editable.zip](../examples/edit-workflow/combined-study-editable.zip),
extract it and inspect its README, license and file hash manifest. Open
`before.blend` to start from the observed partitions, or `after.blend` to inspect
the committed ownership/edit result. The archive also includes `after.glb`, its
identity sidecar and the three actual CPU renders used for the comparison.

In Blender's Text Editor, open the archive's standalone `blender_edit.py` (or
the same file from the repository) and choose
**Run Script** (`Alt+P`). In the 3D View, show the sidebar (`N`) and choose
**AlvenX**. Repeat registration when opening a fresh Blender process. A `.blend`
preserves the observed meshes, source-face attributes and edit journal; the panel
code is supplied separately rather than embedded as automatically executed code.

For a new prepared bundle, open an empty document and use **Import surface bundle**
to select `surface-scene.json`. The importer checks paths, part SHA256 values,
unique region identities, metre units, Y-up input and triangular colored surfaces
before preparing the review document. The viewport converts the world to Blender
Z-up. **Environment + unassigned** contains both environmental geometry and faces
that the initial partition could not confidently assign to an object.

## Review ownership and make an edit

1. Select one imported source mesh, enter Edit Mode and use Face Select. Select a
   nonempty proper subset of faces. Choose **Destination** and click **Assign
   selected faces**. The destination can be another observed region or environment.
   Keep one source object in Edit Mode. Selected observed faces change owner while
   their world position, per-corner colors, materials and source-face identity remain.
2. Return to Object Mode, select an observed object region and enter translation in
   **metres along Blender world X/Y/Z** and rotation in **degrees around world Z**.
   **Apply metre / degree edit** rotates about the current geometry bounds center
   and applies the world translation. Translation is not in the input Y-up frame.
3. Use **Undo** / **Redo** to inspect the native state changes. Save with **Save
   editable .blend** and reopen that document to continue editing.
4. Use **Export GLB + identity record**. The GLB uses glTF Y-up and carries region
   identity/provenance extras. The `.identity.json` sidecar binds the GLB SHA256 to
   source lineage and journal. Retain the `.blend`: its native face attributes are
   authoritative. Sidecar face IDs use native face order; they do not restore
   per-face identity automatically after GLB reimport.

Object-level hidden regions are included during export and their visibility and
selection are restored afterward. Make excluded or blocked collections available
in the active view layer before exporting. Caught sidecar-write or pair-rename
errors restore a prior valid pair. A hard process termination between the two file
replacements is not an atomic publication; compare the sidecar asset hash before
using a package and retain the independently saved native document.

## Reproduce the measured six-scene task suite

Run from the repository root in PowerShell. Executable variables below are
placeholders for already prepared software; no installation is performed here.
If S0 results already exist, reuse their verified `surface/` bundles and skip the
first two commands. Otherwise reproduce [S0](DEVELOPMENT-S0.md) first:

```powershell
$Python = 'C:\path\to\venv312\Scripts\python.exe'
$Blender = 'C:\path\to\blender-5.1.2\blender.exe'
& $Python fixtures/make_complex_fixture.py --output '.local\s0-inputs' --width 160 --height 120
& $Python run_development.py --suite '.local\s0-inputs' --output '.local\s0-results'
& $Python run_edit_suite.py --results '.local\s0-results' --output '.local\s1-results' --blender $Blender --timeout 1200
& $Python blender_edit_controls.py --blender $Blender --output '.local\s1-controls'
$Result = Get-Content -LiteralPath '.local\s1-results\edit-suite-results.json' -Raw | ConvertFrom-Json
if ($Result.status -ne 'pass') { throw 'Native edit suite failed; retain the run and inspect its logs.' }
```

Use a new or empty suite output and a new control output directory. The timeout is
finite, positive **seconds per Blender process**. Failures remain in the result
denominator and logs. To run just one case, add `--scene dev-combined-study` to
`run_edit_suite.py`; the default render case is that combined study. Each case has
one background edit process and a fresh native reopen process. Outputs include
before/after `.blend`, `after.glb` + sidecar, task/reopen JSON, expected-state data
and logs; the combined study also produces `before.png`, `after.png` and
`after-reimport.png`.

The scripted task transfers 12 environment faces to the primary region, translates
that region by `(0.25, -0.1, 0)` m, rotates the secondary region by 32 degrees, then
transfers 8 environment faces to the already transformed secondary region. Face
selection uses proximity to the target bounds center and no reference labels.
These fixed selections test preservation, not segmentation/correction quality or
human effort. `TASKS` in [run_edit_suite.py](../run_edit_suite.py) fixes each pair.

Memory telemetry uses existing optional `psutil` (measured version 7.2.2). Without
it, the task still runs and memory is reported as unavailable. The recorded value
is sampled concurrent RSS of Blender and observed descendants every 0.1 s,
not an exact peak or GPU allocation. Runtime and byte hashes can vary across
hosts/builds. No whole-machine network isolation result is claimed by this suite.

Inspect [the published result JSON](../reports/editing-s1/edit-suite-results.json),
[15 native controls](../reports/editing-s1/controls.json) and
[metrics CSV](../reports/editing-s1/metrics.csv) for the measured denominators,
source hashes, process timings and error limits. The oriented-triangle comparator
preserves face multiplicity under a 0.00005 m corner tolerance; geometric matching
does not prove welded vertex/seam identity. Scene holes and object surfaces still
left in environment remain visible and require further ownership/reconstruction
work rather than another export.

## Rebuild the publication figure

The optional [plot_editing.py](../plot_editing.py) uses the existing plotting
environment (`requirements-plots.txt`) and validates render SHA256 before
producing the 2400×1080 PNG, PDF, per-scene CSV and figure manifest.

```powershell
$PlotPython = 'C:\path\to\plot-environment\python.exe'
& $PlotPython plot_editing.py --results '.local\s1-results\edit-suite-results.json' --assets '.local\s1-results\dev-combined-study' --output '.local\s1-figures'
```

Use a new output directory. The tested Anaconda plotting environment used the
process-local `$env:MKL_THREADING_LAYER = 'SEQUENTIAL'` workaround described in
the S0 runbook; it is unrelated to core Blender execution. The original S0 false
failure is separately downloadable and replayable via the
[original-asset instructions](../examples/edit-workflow/README.md).

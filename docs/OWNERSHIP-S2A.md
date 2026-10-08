# From recorded B1 ownership to an editable scene

[中文](OWNERSHIP-S2A.zh-CN.md)

S2a connects the recorded centroid partition to the existing native Blender
workflow. It also diagnoses the remaining partition errors on the same fixed
meshes. Prepare the [S0 CPU environment and inputs](DEVELOPMENT-S0.md) and
[Blender 5.1.2](EDITING-S1.md) first. Core processing uses local files and CPU;
this stage has no model inference or new paid allocation.

## Export the partition that was actually evaluated

From the repository root in PowerShell, use a new output directory:

```powershell
$Python = 'C:\path\to\venv312\Scripts\python.exe'
$Blender = 'C:\path\to\blender-5.1.2\blender.exe'
& $Python export_partition.py --results '.local\s0-results' --published-results '.local\s0-results\development-results.json' --output '.local\b1-bundles'
& $Python run_edit_suite.py --results '.local\b1-bundles' --output '.local\b1-editing' --blender $Blender --timeout 1200
```

`--published-results` selects the recorded result manifest whose hashes must
match the prepared inputs. For an exact replay of the published S0 files, use
`reports/development-s0-results.json`; a freshly reconstructed run uses its own
`development-results.json`, as above. Different Open3D builds may order the
mesh differently. Matching a new run does not establish byte identity with the
published mesh.

The exporter checks mesh, observations, original parts and owner-array hashes.
It does not load reference labels. Each input triangle appears in exactly one
output part, with unchanged geometry and observed color. Per-part index maps
link the Blender source-face inventory back to the original reconstructed mesh.
An independent native import check verifies this mapping in saved documents.

Run the diagnostic and independent native check after preparing S0 and B1:

```powershell
& $Python partition_diagnostics.py --results '.local\s0-results' --suite '.local\s0-input' --published-results '.local\s0-results\development-results.json' --output '.local\s2a-diagnostics'
& $Python verify_partition_import.py --results '.local\s0-results' --published-results '.local\s0-results\development-results.json' --bundles '.local\b1-bundles' --editing '.local\b1-editing' --output '.local\s2a-lineage' --blender $Blender
```

`--suite` is the same generated fixture input directory used for S0. Outputs
must be new directories. Diagnostic per-face `face-diagnostics.npz` arrays contain
reference evidence and must not be used to generate ownership predictions.
Array columns and hashes are documented in `diagnostics-results.json`.
The [measured report](../reports/OWNERSHIP-S2A-2026-10-08.md) includes raw summary
JSON/CSV. Optional figure rebuilding uses Matplotlib 3.10.0:
`python plot_ownership.py --results '.local/s2a-diagnostics/diagnostics-results.json' --output '.local/s2a-figures'`.
Matplotlib is only needed for figure regeneration. Six scene records are required
for this fixed development overview. On the measured Anaconda environment,
`$env:MKL_THREADING_LAYER='SEQUENTIAL'` was needed in that plotting shell.

The exported layout is `<scene>/surface/surface-scene.json`, accepted by the
unchanged S1 panel and task runner. Overlapping boxes produce a separate
**Unresolved (-2)** region. This region uses the existing region transport type,
but its ID, category and manifest state explicitly identify unresolved ownership.
Preserve the whole bundle and its index maps alongside saved native documents.

## Complete an author-operated task

To start immediately, extract the [initial B1 dining package](../examples/ownership-b1/dining-b1-author-task.zip).
It contains `before.blend`, `blender_edit.py`, the complete surface bundle and
a byte-level manifest. Blender 5.1.2 is sufficient for the manual task; Open3D
is used only to rebuild and independently verify the source pipeline.

Open the generated `before.blend` for `dev-same-color-dining` or
`dev-occluded-bookshelf`. Register `blender_edit.py` in Blender's Text Editor
and open the **AlvenX** sidebar, following [the S1 instructions](EDITING-S1.md).
Start with the initial B1 partition, rather than the scripted `after.blend`.

1. Identify a real target omission. In the dining scene inspect **Unresolved (-2)**;
   for the shelf also inspect the environment. Inspect the geometry before deciding
   its owner. A diagnostic label is evaluation evidence, not an automatic repair.
2. Select a proper subset of source faces in Edit Mode and assign it to the target.
   Record selections, corrections and mistakes as they happen.
3. Return to Object Mode, translate the target by a recorded metre offset, rotate
   another object, and exercise Undo/Redo through the actual interface.
4. Save, close and reopen the native document; export GLB and keep its identity
   sidecar. Check the corrected object, unresolved remainder and other geometry.

Record the operator, input hashes, task start/end, active correction time,
selection/assignment actions, undo operations, failure/recovery and final files.
Score final ownership only after the task when reference labels are available.
An unperformed task stays **pending**. Script runtime, replay count and generated
screenshots do not measure author effort or usability.

The automated S1 task transfers a fixed 12/8-face selection and tests preservation.
Those selections are not a claim that the object boundary was corrected. The
diagnostic reference also has the S0 nearest-surface visibility/normal limitations;
unreconstructed surfaces are outside the partition-error denominator.

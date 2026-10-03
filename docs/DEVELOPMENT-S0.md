# Reproduce the six-scene S0 development experiment

[中文](DEVELOPMENT-S0.zh-CN.md) · [Measured report](../reports/DEVELOPMENT-S0-2026-10-04.md)

Run from this repository root. These original MIT inputs are generated locally;
no scan download, account, GPU inference or Blender is needed for this comparison.
Prepare Python 3.12 and the wheelhouse using [the existing environment guide](REPRODUCING.md).
The tested core environment is Python 3.12.14, NumPy 2.3.5, Pillow 12.3.0 and
Open3D 0.19.0 on Windows. The algorithms execute on CPU.

```powershell
$Python = 'C:\path\to\venv312\Scripts\python.exe'
& $Python -m unittest discover -s tests -v
& $Python fixtures/make_complex_fixture.py --output '.local\s0-inputs' --width 160 --height 120
& $Python offline_run.py --script run_development.py --report '.local\s0-network-guard.json' -- --suite '.local\s0-inputs' --output '.local\s0-results'
& $Python reevaluate_development.py --results '.local\s0-results' --suite '.local\s0-inputs' --output '.local\s0-sensitivity'
```

Use a new or empty output directory for each experiment and sensitivity run.
Failed tasks remain in the JSON denominators. Do not reuse outputs as if a failed
attempt had succeeded. The generator rejects links/junctions and protects foreign
nonempty directories; a prior suite from this generator may be regenerated.

`suite-manifest.json` fixes scene IDs, development split, seeds, generator/config
hashes, frame counts and stress assumptions. Each scene has `scan/` (measured
RGB-D, calibration, measured poses and box priors) and `reference/` (physical
geometry, independent instance labels, true poses and ray hits). `run_development.py`
computes all three partitions before loading the reference labels.

Results include the full fixed mesh, each method's face labels and area metrics,
and `development-results.json` / `development-metrics.csv`. Scene, surface manifest,
mesh, reference and label/metric hashes anchor the sensitivity replay. Absolute
paths in intermediate local manifests identify this run; newly generated manifests
use the new machine's paths. Times and OS details naturally vary across hosts.
Different Open3D builds may change TSDF vertex ordering; compare area results and
document the build rather than requiring another host's mesh byte identity.

The sensitivity replay keeps the saved mesh and labels fixed, evaluates distance
tolerances 0.04 / 0.08 / 0.12 m and 7 / 49 area samples per triangle, and checks the
0.08 m / 7-sample scores against the original run. It does not tune or rerun a
partition method. The 2 mm competing-label ambiguity margin remains fixed.

## Rebuild the figures

Plotting is optional and uses a separate environment; core reproduction has no
Matplotlib dependency. The measured figure environment used Python 3.13.5,
Matplotlib 3.10.0 and NumPy 2.1.3 (`requirements-plots.txt`).

```powershell
$PlotPython = 'C:\path\to\plot-environment\python.exe'
& $PlotPython plot_development.py --results '.local\s0-results\development-results.json' --suite '.local\s0-inputs' --output '.local\s0-figures'
```

On the tested Anaconda installation, Intel's parallel MKL runtime DLL was missing.
The recorded workaround was `$env:MKL_THREADING_LAYER = 'SEQUENTIAL'` for that
plot process; no library was installed or global configuration changed. The figure
manifest records the active value. Other environments do not require this setting.
The scene overview renders the independent authored reference, while the chart
shows partition metrics on reconstructed surfaces. Neither is an editable-scene
product acceptance result.

## Network and next-stage boundaries

`offline_run.py` denies CPython outbound socket and DNS operations and tests that
denial. It does not isolate native libraries, child processes or the whole machine.
Whole-machine disconnected execution remains a separate acceptance item.
The full [evaluation protocol](NEXT-STAGE-PROTOCOL.md) additionally requires
held-out scenes, a defensible visibility rule, real-room evidence and complete
editing tasks. S0 only establishes this six-scene development comparison.

# Table0 rigid-track feasibility

The bounded 2026-10-25 tea-room experiment ended **INCONCLUSIVE**. The same
eight frames and sixteen predeclared rigid correspondences were used to compare
the original camera and two previously fitted alternatives. The existing
[Table0 geometry rejection](TABLE0-DIAGNOSIS.md) remains in force.

Two agent annotators independently recorded visibility and position against a
shared, fixed catalogue. The second annotator received the catalogue and input
images. An observation qualified when both annotators agreed on the visible
physical point and their location disks overlapped. A track needed at least
three qualifying views and a connection to an unchanged anchor frame. All
candidates and missing pair scores were retained.

| Modified frame | Clear A | Clear B | Anchor-connected tracks | Required | Occupied grid cells of 12 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 46 | 3 | 4 | 0 | 6 | 0 |
| 47 | 3 | 3 | 1 | 6 | 1 |
| 48 | 8 | 6 | 4 | 6 | 2 |
| 49 | 6 | 5 | 3 | 6 | 1 |

Only four of the sixteen proposed tracks qualified. Of 28 predeclared frame
pairs, 15 had numerical support and none met the pair gate of six tracks
across two regions. Nine of the 16 primary modified-to-anchor pairs had scores.
The sign of the candidate-minus-original median Sampson residual changed under
annotation perturbations for seven normal-candidate pairs and six
balanced-candidate pairs. Three of the six unchanged-anchor controls had
observations and returned zero change. The three empty controls stayed unscored.
The 256 seeded perturbations measure sensitivity to the stated image-coordinate
uncertainty. Physical dimensions and pose accuracy would require independent
reference measurements.

The result bounds what this fixed sample can establish. Improving Table0 pose
on the available observations requires more supported views or an independent
metric reference. The experiment ran in an existing local Python 3.12.14
environment and made no change to the adopted scene.
[Machine-readable aggregate](../../reports/table0-rigid-tracks-2026-10-25.json)

## Code and replay

The evaluator handles consensus, anchor support, input verification and
shared-perturbation scoring. Its synthetic regression tests run from the
repository root after installing `requirements.txt`.

```powershell
python -B -m unittest discover -s tests -p 'test_audit_rigid_tracks.py' -v
python -B -m unittest discover -s tests -p 'test_rgb_pose_check.py' -v
python -B audit_rigid_tracks.py --help
```

A holder of the frozen private inputs can run
`python -B audit_rigid_tracks.py evaluate --seal <seal.json> --output <new-directory>`.
The seal includes the dataset, annotation, protocol and evaluator hashes.
The public aggregate omits the original images, camera records, individual
annotations and location samples. The raw replay depends on those inputs.
[Evaluator](../../audit_rigid_tracks.py) · [Tests](../../tests/test_audit_rigid_tracks.py) · [Scan source terms](../../THIRD_PARTY_NOTICES.md)

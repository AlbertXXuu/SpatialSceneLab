"""Read-only recheck of frozen S0 native assets; no Blender save/export."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import bpy

parser = argparse.ArgumentParser()
parser.add_argument('--project', type=Path, required=True)
parser.add_argument('--assets', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args(sys.argv[sys.argv.index('--') + 1:])
sys.path.insert(0, str(args.project.resolve()))
import blender_scene

paths = [args.assets / name for name in ['before.blend', 'before.glb', 'after.blend', 'after.glb', 'blender-verification.json']]
def hashes():
    return {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
initial = hashes()
started = time.perf_counter()
result = {'blender': bpy.app.version_string, 'assets_sha256': initial, 'checks': {}}
for stage in ['before', 'after']:
    bpy.ops.wm.open_mainfile(filepath=str(args.assets / (stage + '.blend')))
    original = blender_scene.world_geometry()
    returned = blender_scene.reopen_glb(args.assets / (stage + '.glb'))
    result['checks'][stage] = blender_scene.compare_geometry(original, returned)
    if stage == 'before':
        result['negative_controls'] = blender_scene.verifier_negative_controls(original)
result['elapsed_seconds'] = time.perf_counter() - started
result['assets_unchanged'] = initial == hashes()
args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n', encoding='utf-8')
print('ALVENX_READONLY_RECHECK ' + json.dumps({stage: result['checks'][stage]['status'] for stage in ['before', 'after']}))

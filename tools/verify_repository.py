"""Check packaged records and syntax without training or importing pilot scripts."""
import ast
import hashlib
import json
import os
from pathlib import Path

from summarize_results import ROOT, recovery_table, segmentation_table, require


def readable(path):
    # Windows legacy artifact paths exceed MAX_PATH in some checkout locations.
    return Path('\\\\?\\'+str(path.resolve())) if os.name == 'nt' else path


def main():
    manifest = json.loads((ROOT/'docs/source_manifest.json').read_text())
    edits = json.loads((ROOT/'docs/packaging_edits.json').read_text())
    for entry in manifest:
        data = readable(ROOT/entry['path']).read_bytes()
        expected = edits.get(entry['path'], {}).get('packaged_sha256', entry['sha256'])
        require(hashlib.sha256(data).hexdigest() == expected, f'Changed imported file: {entry["path"]}')
    count = 0
    for path in readable(ROOT).rglob('*.py'):
        if '.git' not in path.parts:
            ast.parse(path.read_text(encoding='utf-8-sig'), filename=str(path))
            count += 1
    recovery_table()
    segmentation_table()
    print(f'PASS: {len(manifest)} imported files verified; {len(edits)} declared packaging edits; {count} Python files parsed.')
    print('PASS: saved recovery and segmentation tables and validation-based choices agree.')
    print('No training/model imports, image inference or dataset access.')


if __name__ == '__main__':
    main()

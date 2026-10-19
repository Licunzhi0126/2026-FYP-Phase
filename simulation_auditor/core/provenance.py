"""Hashes and traceability of the source of every report."""
import hashlib
import json
from pathlib import Path

def sha256_file(path):
    path = Path(path)
    if not path.is_file(): return None
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def json_write(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf8')

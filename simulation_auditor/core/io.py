"""Filesystem/ZIP CSV access; no biological assumptions here."""
import io
from pathlib import Path
import zipfile
import pandas as pd

class CsvStore:
    def __init__(self, path):
        self.path = Path(path).resolve()
        if not self.path.exists():
            raise FileNotFoundError(self.path)
        if self.path.is_file() and not zipfile.is_zipfile(self.path):
            raise ValueError(f"Expected a ZIP or directory: {self.path}")
        self.zip = zipfile.ZipFile(self.path) if self.path.is_file() else None
        self.names = self.zip.namelist() if self.zip else [
            f.relative_to(self.path).as_posix() for f in self.path.rglob('*.csv')
        ]

    def locate(self, suffix):
        suffix = suffix.strip('/')
        matches = [s for s in self.names if s == suffix or s.endswith('/' + suffix)]
        matches = [s for s in matches if not s.startswith('__MACOSX/') and '/._' not in s]
        if not matches:
            raise FileNotFoundError(f"Missing {suffix!r} in {self.path}")
        # Avoid ambiguous paths selecting silently.
        matches.sort(key=lambda s: (len(s), s))
        if len(matches) > 1 and len(matches[0]) == len(matches[1]):
            raise ValueError(f"Ambiguous CSV path {suffix}: {matches}")
        return matches[0]

    def read(self, suffix):
        name = self.locate(suffix)
        if self.zip:
            return pd.read_csv(io.BytesIO(self.zip.read(name)))
        return pd.read_csv(self.path / name)

    def close(self):
        if self.zip: self.zip.close()

    def __enter__(self): return self
    def __exit__(self, *_): self.close()

"""
Dicom utilities, like finding out if x-ray is frontal or lateral
"""

from lib.utils.patterns import singleton


import json
from pathlib import Path


@singleton
class XRayJsonClassifier:
    def __init__(self, root_data="data") -> None:
        self.json_path = Path(root_data) / "xrays-view-annotation.json"
        self.load_json()

    def load_json(self):
        try:
            with open(self.json_path) as f:
                self.annotations = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Annotation file not found at {self.json_path}")
        except json.JSONDecodeError:
            raise ValueError(f"Error decoding JSON from {self.json_path}")

    def _preprocess_path(self, path: str | Path) -> str:
        path = Path(path)
        while str(path) not in self.annotations and len(path.parts) > 1:
            path = Path(*path.parts[1:])
        if len(path.parts) == 0:
            raise ValueError("""Provided path couldn't be found at the
                             annotations""")
        return str(path)

    def classify(self, path: str | Path) -> str:
        # original_path = path
        # print(original_path)
        path = self._preprocess_path(path=path)
        return self.annotations[path]

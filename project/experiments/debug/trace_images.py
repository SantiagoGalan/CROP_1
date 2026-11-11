"""Registro de imagenes de los modelos CROP en ejecucion eager, sin modificarlos."""

from pathlib import Path
import sys

import numpy as np


class ImageTrace:
    """Captura cambios de tensores de imagen en filter, decode y correcciones."""

    def __init__(self, model, image_shape=(28, 28)):
        self.model = model
        self.image_shape = image_shape
        self.records = []
        self._frames = {}
        self._source = 0

    def _image(self, value):
        if not isinstance(value, np.ndarray) and not hasattr(value, "numpy"):
            return None
        array = np.asarray(value)
        if array.size != np.prod(self.image_shape):
            return None
        return array.reshape(self.image_shape).copy()

    def _record(self, frame):
        state = self._frames[id(frame)]
        values = dict(frame.f_locals)
        if frame.f_code.co_name == "decode":
            values.update({
                "source1_estimation": self.model.source1_estimation,
                "source2_estimation": self.model.source2_estimation,
            })
        for name, value in values.items():
            if name.startswith("_"):
                continue
            image = self._image(value)
            if image is None:
                continue
            previous = state["last"].get(name)
            if previous is not None and np.array_equal(previous, image, equal_nan=True):
                continue
            state["last"][name] = image
            self.records.append({
                "iteration": self.model.iteration + 1,
                "source": state["source"],
                "function": frame.f_code.co_name,
                "variable": name,
                "file": frame.f_code.co_filename,
                "line": frame.f_lineno,
                "image": image,
            })

    def _trace(self, frame, event, arg):
        if event == "call":
            path = Path(frame.f_code.co_filename)
            if path.parent.name != "models" or path.parent.parent.name != "CROP":
                return None
            if frame.f_code.co_name not in {
                "filter", "decode", "subtract_saturation_border_mean", "step", "body"
            }:
                return None
            if frame.f_code.co_name == "filter":
                self._source += 1
            self._frames[id(frame)] = {
                "source": (self._source - 1) % 2 + 1 if self._source else 0,
                "last": {},
            }
            return self._trace
        if id(frame) in self._frames:
            if event in {"line", "return"}:
                self._record(frame)
            if event == "return":
                image = self._image(arg)
                if image is not None:
                    self.records.append({
                        "iteration": self.model.iteration + 1,
                        "source": self._frames[id(frame)]["source"],
                        "function": frame.f_code.co_name,
                        "variable": "return",
                        "file": frame.f_code.co_filename,
                        "line": frame.f_lineno,
                        "image": image,
                    })
                del self._frames[id(frame)]
        return self._trace

    def __enter__(self):
        self._previous = sys.gettrace()
        sys.settrace(self._trace)
        return self

    def __exit__(self, *exc):
        sys.settrace(self._previous)
        self._frames.clear()

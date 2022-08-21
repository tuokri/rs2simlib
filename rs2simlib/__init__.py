from pathlib import Path as _Path

from . import dataio
from . import drag
from . import fast
from . import models

__all__ = [
    "version",
    "__version__",
    "__version_tuple__",
    "dataio",
    "drag",
    "fast",
    "models",
]

_version = {}  # type: ignore[var-annotated]
with open(f"{_Path(__file__).parent}/_version.py") as _fp:
    exec(_fp.read(), _version)

__version__ = _version["__version__"]
__version_tuple__ = _version["__version_tuple__"]
version = __version__

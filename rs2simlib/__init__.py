from pathlib import Path as _Path

from . import fast

__all__ = [
    "version",
    "__version__",
    "__version_tuple__",
    "fast",
]

_version = {}
with open(f"{_Path(__file__).parent}/_version.py") as _fp:
    exec(_fp.read(), _version)

__version__ = _version["__version__"]
__version_tuple__ = _version["__version_tuple__"]
version = __version__

"""
blobfigy: Python bindings for the blobfig binary configuration format.

Example:
    import blobfigy
    import numpy as np

    # Create a config
    config = {
        "version": 1,
        "model": blobfigy.File.from_path("model.tflite"),
        "mean": blobfigy.Array.from_numpy(np.array([0.485, 0.456, 0.406], dtype=np.float32)),
    }

    # Serialize
    data = blobfigy.to_bytes(config)

    # Parse
    parsed = blobfigy.parse(data)
    mean = parsed["mean"].to_numpy()
"""

import importlib.metadata

from .array import Array
from .file import File
from ._blobfigy import parse as _parse, to_bytes as _to_bytes
from ._blobfigy import Array as _Array, File as _File

# Value types that blobfig supports
Value = dict[str, "Value"] | list["Value"] | int | float | str | bool | Array | File
_RustValue = dict[str, "_RustValue"] | list["_RustValue"] | int | float | str | bool | _Array | _File


def parse(data: bytes) -> Value:
    """
    Parse blobfig bytes into a Python object.

    Args:
        data: Raw blobfig bytes

    Returns:
        Parsed value. Arrays and Files are wrapped in Python classes.

    Raises:
        ValueError: If parsing fails
    """
    return _wrap_value(_parse(data))


def to_bytes(obj: Value) -> bytes:
    """
    Serialize a Python object to blobfig bytes.

    Args:
        obj: Value to serialize (dict, list, int, float, str, bool, Array, or File)

    Returns:
        Serialized bytes

    Raises:
        ValueError: If serialization fails (e.g., unsupported type or key contains '/')
    """
    return _to_bytes(_unwrap_value(obj))


def _wrap_value(val: _RustValue) -> Value:
    """Wrap Rust types in Python wrappers."""
    if isinstance(val, _Array):
        return Array._from_inner(val)
    elif isinstance(val, _File):
        return File._from_inner(val)
    elif isinstance(val, dict):
        return {k: _wrap_value(v) for k, v in val.items()}
    elif isinstance(val, list):
        return [_wrap_value(v) for v in val]
    else:
        return val


def _unwrap_value(val: Value) -> _RustValue:
    """Unwrap Python wrappers to Rust types."""
    if isinstance(val, Array):
        return val._inner
    elif isinstance(val, File):
        return val._inner
    elif isinstance(val, dict):
        return {k: _unwrap_value(v) for k, v in val.items()}
    elif isinstance(val, list):
        return [_unwrap_value(v) for v in val]
    else:
        return val


__version__ = importlib.metadata.version("blobfigy")

__all__ = [
    "Array",
    "File",
    "Value",
    "parse",
    "to_bytes",
    "__version__",
]

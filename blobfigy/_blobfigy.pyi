"""Type stubs for the blobfigy native module."""

import numpy as np
import numpy.typing as npt

# Supported numpy dtypes
NumpyDType = (
    np.uint8 | np.int8 | np.uint16 | np.int16 |
    np.uint32 | np.int32 | np.uint64 | np.int64 |
    np.float32 | np.float64
)

# Value type alias
Value = dict[str, "Value"] | list["Value"] | int | float | str | bool | "Array" | "File"


class Array:
    """Typed array with shape information."""

    def __init__(self, dtype: str, shape: list[int], data: bytes) -> None:
        """
        Create an array.

        Args:
            dtype: Data type ("u8", "i8", "u16", "i16", "u32", "i32", "u64", "i64", "f32", "f64")
            shape: Shape as list of dimensions
            data: Raw bytes in little-endian format
        """
        ...

    @property
    def dtype(self) -> str:
        """Data type string."""
        ...

    @property
    def shape(self) -> list[int]:
        """Shape as list of dimensions."""
        ...

    @property
    def data(self) -> bytes:
        """Raw bytes."""
        ...

    def to_numpy(self) -> npt.NDArray[NumpyDType]:
        """Convert to numpy array."""
        ...

    def __repr__(self) -> str: ...


class File:
    """Embedded file blob with mimetype."""

    def __init__(self, mimetype: str, data: bytes) -> None:
        """
        Create a file.

        Args:
            mimetype: MIME type string
            data: File contents as bytes
        """
        ...

    @property
    def mimetype(self) -> str:
        """MIME type string."""
        ...

    @property
    def data(self) -> bytes:
        """File contents."""
        ...

    def __repr__(self) -> str: ...


def parse(data: bytes) -> Value:
    """
    Parse blobfig bytes into a Python object.

    Args:
        data: Raw blobfig bytes

    Returns:
        Parsed value (dict, list, int, float, str, bool, Array, or File)

    Raises:
        ValueError: If parsing fails
    """
    ...


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
    ...

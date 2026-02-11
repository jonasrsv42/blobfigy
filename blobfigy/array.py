"""Array wrapper for blobfig arrays."""

import numpy as np
import numpy.typing as npt

from ._blobfigy import Array as _Array

# Supported numpy dtypes
NumpyDType = (
    np.uint8 | np.int8 | np.uint16 | np.int16 |
    np.uint32 | np.int32 | np.uint64 | np.int64 |
    np.float32 | np.float64
)


class Array:
    """
    Typed array with shape information.

    Can be created from numpy arrays or raw bytes.

    Example:
        # From numpy
        arr = Array.from_numpy(np.array([1.0, 2.0, 3.0], dtype=np.float32))

        # From raw bytes
        arr = Array("f32", [3], b"\\x00\\x00\\x80?\\x00\\x00\\x00@\\x00\\x00@@")

        # Convert to numpy
        np_arr = arr.to_numpy()
    """

    def __init__(self, dtype: str, shape: list[int], data: bytes) -> None:
        """
        Create an array from raw bytes.

        Args:
            dtype: Data type ("u8", "i8", "u16", "i16", "u32", "i32", "u64", "i64", "f32", "f64")
            shape: Shape as list of dimensions
            data: Raw bytes in little-endian format
        """
        self._inner = _Array(dtype, shape, data)

    @classmethod
    def from_numpy(cls, arr: npt.NDArray[NumpyDType]) -> "Array":
        """
        Create an Array from a numpy array.

        Args:
            arr: Numpy array (will be converted to little-endian contiguous layout)

        Returns:
            Array wrapping the data

        Raises:
            ValueError: If dtype is not supported
        """
        dtype_map = {
            np.dtype("uint8"): "u8",
            np.dtype("int8"): "i8",
            np.dtype("uint16"): "u16",
            np.dtype("int16"): "i16",
            np.dtype("uint32"): "u32",
            np.dtype("int32"): "i32",
            np.dtype("uint64"): "u64",
            np.dtype("int64"): "i64",
            np.dtype("float32"): "f32",
            np.dtype("float64"): "f64",
        }
        dtype_str = dtype_map.get(arr.dtype)
        if dtype_str is None:
            raise ValueError(f"Unsupported dtype: {arr.dtype}")

        # Ensure contiguous little-endian layout
        arr = np.ascontiguousarray(arr)
        if arr.dtype.byteorder == ">":
            arr = arr.byteswap().view(arr.dtype.newbyteorder("<"))

        shape = list(arr.shape)
        data = arr.tobytes()
        return cls(dtype_str, shape, data)

    @classmethod
    def _from_inner(cls, inner: _Array) -> "Array":
        """Create from inner Rust type (internal use)."""
        instance = cls.__new__(cls)
        instance._inner = inner
        return instance

    @property
    def dtype(self) -> str:
        """Data type string."""
        return self._inner.dtype

    @property
    def shape(self) -> list[int]:
        """Shape as list of dimensions."""
        return self._inner.shape

    @property
    def data(self) -> bytes:
        """Raw bytes."""
        return self._inner.data

    def to_numpy(self) -> npt.NDArray[NumpyDType]:
        """Convert to numpy array."""
        return self._inner.to_numpy()

    def __repr__(self) -> str:
        return f"Array(dtype='{self.dtype}', shape={self.shape})"

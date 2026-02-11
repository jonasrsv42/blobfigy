"""File wrapper for blobfig embedded files."""

from pathlib import Path

from ._blobfigy import File as _File


class File:
    """
    Embedded file blob with mimetype.

    Example:
        # From bytes
        f = File("text/plain", b"Hello, world!")

        # From file path
        f = File.from_path("model.tflite", mimetype="application/x-tflite")

        # Access data
        print(f.mimetype)  # "text/plain"
        print(f.data)      # b"Hello, world!"
    """

    def __init__(self, mimetype: str, data: bytes) -> None:
        """
        Create a file from bytes.

        Args:
            mimetype: MIME type string
            data: File contents as bytes
        """
        self._inner = _File(mimetype, data)

    @classmethod
    def from_path(cls, path: str | Path, mimetype: str | None = None) -> "File":
        """
        Create a File from a file path.

        Args:
            path: Path to the file
            mimetype: MIME type (guessed from extension if not provided)

        Returns:
            File wrapping the contents
        """
        path = Path(path)
        data = path.read_bytes()

        if mimetype is None:
            mimetype = _guess_mimetype(path.suffix)

        return cls(mimetype, data)

    @classmethod
    def _from_inner(cls, inner: _File) -> "File":
        """Create from inner Rust type (internal use)."""
        instance = cls.__new__(cls)
        instance._inner = inner
        return instance

    @property
    def mimetype(self) -> str:
        """MIME type string."""
        return self._inner.mimetype

    @property
    def data(self) -> bytes:
        """File contents."""
        return self._inner.data

    def save(self, path: str | Path) -> None:
        """
        Save file contents to a path.

        Args:
            path: Destination path
        """
        Path(path).write_bytes(self.data)

    def __repr__(self) -> str:
        return f"File(mimetype='{self.mimetype}', size={len(self.data)})"


def _guess_mimetype(suffix: str) -> str:
    """Guess mimetype from file suffix."""
    mimetypes = {
        ".txt": "text/plain",
        ".json": "application/json",
        ".tflite": "application/x-tflite",
        ".onnx": "application/x-onnx",
        ".pt": "application/x-pytorch",
        ".pth": "application/x-pytorch",
        ".bin": "application/octet-stream",
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
    }
    return mimetypes.get(suffix.lower(), "application/octet-stream")

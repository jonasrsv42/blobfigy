"""Test roundtrip serialization/deserialization."""

import numpy as np
import pytest

import blobfigy


class TestPrimitives:
    def test_bool_true(self) -> None:
        data = blobfigy.to_bytes(True)
        assert blobfigy.parse(data) is True

    def test_bool_false(self) -> None:
        data = blobfigy.to_bytes(False)
        assert blobfigy.parse(data) is False

    def test_int(self) -> None:
        data = blobfigy.to_bytes(42)
        assert blobfigy.parse(data) == 42

    def test_int_negative(self) -> None:
        data = blobfigy.to_bytes(-123)
        assert blobfigy.parse(data) == -123

    def test_float(self) -> None:
        data = blobfigy.to_bytes(3.14159)
        assert abs(blobfigy.parse(data) - 3.14159) < 1e-10

    def test_string(self) -> None:
        data = blobfigy.to_bytes("hello world")
        assert blobfigy.parse(data) == "hello world"

    def test_string_unicode(self) -> None:
        data = blobfigy.to_bytes("hello 世界")
        assert blobfigy.parse(data) == "hello 世界"


class TestCollections:
    def test_list(self) -> None:
        original = [1, 2, 3, "four", True]
        data = blobfigy.to_bytes(original)
        parsed = blobfigy.parse(data)
        assert parsed == original

    def test_dict(self) -> None:
        original = {"name": "test", "version": 1, "enabled": True}
        data = blobfigy.to_bytes(original)
        parsed = blobfigy.parse(data)
        assert parsed == original

    def test_nested(self) -> None:
        original = {
            "config": {"threshold": 0.5, "enabled": True},
            "items": [1, 2, 3],
        }
        data = blobfigy.to_bytes(original)
        parsed = blobfigy.parse(data)
        assert parsed == original


class TestArray:
    def test_array_f32(self) -> None:
        arr = blobfigy.Array.from_numpy(np.array([1.0, 2.0, 3.0], dtype=np.float32))
        data = blobfigy.to_bytes(arr)
        parsed = blobfigy.parse(data)

        assert isinstance(parsed, blobfigy.Array)
        assert parsed.dtype == "f32"
        assert parsed.shape == [3]

        np_arr = parsed.to_numpy()
        np.testing.assert_array_almost_equal(np_arr, [1.0, 2.0, 3.0])

    def test_array_f64(self) -> None:
        arr = blobfigy.Array.from_numpy(np.array([1.0, 2.0, 3.0], dtype=np.float64))
        data = blobfigy.to_bytes(arr)
        parsed = blobfigy.parse(data)

        assert parsed.dtype == "f64"
        np.testing.assert_array_almost_equal(parsed.to_numpy(), [1.0, 2.0, 3.0])

    def test_array_i32(self) -> None:
        arr = blobfigy.Array.from_numpy(np.array([1, 2, 3], dtype=np.int32))
        data = blobfigy.to_bytes(arr)
        parsed = blobfigy.parse(data)

        assert parsed.dtype == "i32"
        np.testing.assert_array_equal(parsed.to_numpy(), [1, 2, 3])

    def test_array_2d(self) -> None:
        original = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        arr = blobfigy.Array.from_numpy(original)
        data = blobfigy.to_bytes(arr)
        parsed = blobfigy.parse(data)

        assert parsed.shape == [2, 2]
        np.testing.assert_array_almost_equal(parsed.to_numpy(), original)


class TestFile:
    def test_file_bytes(self) -> None:
        f = blobfigy.File("text/plain", b"Hello, world!")
        data = blobfigy.to_bytes(f)
        parsed = blobfigy.parse(data)

        assert isinstance(parsed, blobfigy.File)
        assert parsed.mimetype == "text/plain"
        assert parsed.data == b"Hello, world!"

    def test_file_binary(self) -> None:
        content = bytes(range(256))
        f = blobfigy.File("application/octet-stream", content)
        data = blobfigy.to_bytes(f)
        parsed = blobfigy.parse(data)

        assert parsed.data == content


class TestMLConfig:
    def test_ml_artifact(self) -> None:
        """Test a realistic ML config structure."""
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        model_bytes = b"\xde\xad\xbe\xef" * 100

        config = {
            "version": 1,
            "model_type": "image-classifier",
            "model": blobfigy.File("application/x-tflite", model_bytes),
            "preprocessing": {
                "mean": blobfigy.Array.from_numpy(mean),
                "std": blobfigy.Array.from_numpy(std),
                "input_shape": [224, 224, 3],
            },
            "labels": ["cat", "dog", "bird"],
        }

        data = blobfigy.to_bytes(config)
        parsed = blobfigy.parse(data)

        assert parsed["version"] == 1
        assert parsed["model_type"] == "image-classifier"
        assert parsed["model"].mimetype == "application/x-tflite"
        assert len(parsed["model"].data) == 400

        np.testing.assert_array_almost_equal(
            parsed["preprocessing"]["mean"].to_numpy(), mean
        )
        assert parsed["labels"] == ["cat", "dog", "bird"]


class TestErrors:
    def test_key_with_slash_rejected(self) -> None:
        with pytest.raises(ValueError, match="/"):
            blobfigy.to_bytes({"invalid/key": 1})

    def test_none_rejected(self) -> None:
        with pytest.raises(ValueError, match="None"):
            blobfigy.to_bytes(None)

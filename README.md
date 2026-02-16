# blobfigy

Python bindings for [blobfig](https://github.com/jonasrsv42/blobfig) -- a binary format for bundling ML models, arrays, and configuration into a single blob.

## Why

Deploying an ML model usually means shipping a model file, a config file, preprocessing parameters, label maps, and hoping they all stay in sync. blobfig packs everything into one binary blob. blobfigy lets you build and read those blobs from Python.

## Install

```bash
pip install blobfigy
```

### Build from source

Requires [Rust](https://rustup.rs/) and [maturin](https://www.maturin.rs/):

```bash
pip install maturin
git clone <repo-url>
cd blobfigy
maturin develop
```

## Quick start

```python
import blobfigy
import numpy as np

# Bundle a model with its config
config = {
    "version": 1,
    "model_type": "image-classifier",
    "model": blobfigy.File.from_path("model.tflite"),
    "preprocessing": {
        "mean": blobfigy.Array.from_numpy(
            np.array([0.485, 0.456, 0.406], dtype=np.float32)
        ),
        "std": blobfigy.Array.from_numpy(
            np.array([0.229, 0.224, 0.225], dtype=np.float32)
        ),
        "input_shape": [224, 224, 3],
    },
    "labels": ["cat", "dog", "bird"],
}

# Serialize to bytes
data = blobfigy.to_bytes(config)

# Write to disk
with open("model.blobfig", "wb") as f:
    f.write(data)

# Read it back
with open("model.blobfig", "rb") as f:
    parsed = blobfigy.parse(f.read())

mean = parsed["preprocessing"]["mean"].to_numpy()  # np.array([0.485, 0.456, 0.406])
parsed["model"].save("restored_model.tflite")       # write model back to disk
```

## API

### `to_bytes(obj) -> bytes`

Serialize a Python object to blobfig binary format.

```python
data = blobfigy.to_bytes({"threshold": 0.5, "enabled": True})
```

### `parse(data) -> Value`

Deserialize blobfig bytes back into Python objects.

```python
obj = blobfigy.parse(data)
```

### `Array`

Typed multi-dimensional array with shape and dtype metadata.

```python
# From numpy
arr = blobfigy.Array.from_numpy(np.zeros((224, 224, 3), dtype=np.float32))

# From raw bytes
arr = blobfigy.Array("f32", [3], raw_bytes)

# Properties
arr.dtype  # "f32"
arr.shape  # [224, 224, 3]
arr.data   # raw bytes

# Back to numpy
np_arr = arr.to_numpy()
```

Supported dtypes: `u8`, `i8`, `u16`, `i16`, `u32`, `i32`, `u64`, `i64`, `f32`, `f64`

### `File`

Embedded binary file with MIME type.

```python
# From disk (MIME type guessed from extension)
f = blobfigy.File.from_path("model.onnx")

# From bytes
f = blobfigy.File("application/x-tflite", model_bytes)

# Properties
f.mimetype  # "application/x-tflite"
f.data      # raw bytes

# Save to disk
f.save("output.onnx")
```

Auto-detected MIME types: `.tflite`, `.onnx`, `.pt`/`.pth`, `.json`, `.txt`, `.png`, `.jpg`/`.jpeg`, `.bin`

### Supported types

The `Value` type represents everything blobfig can store:

```
dict[str, Value] | list[Value] | int | float | str | bool | Array | File
```

Dictionary keys cannot contain `/`. `None` is not supported.

## Development

```bash
pip install -e ".[dev]"
maturin develop
pytest
```

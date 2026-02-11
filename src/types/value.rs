//! Conversion between Python objects and blobfig Values

use super::array::PyArray;
use super::file::PyFile;
use blobfig::{Array, DType, File, Value, ValueView};
use numpy::{PyArray1, PyArrayMethods, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyBytes, PyDict, PyFloat, PyInt, PyList, PyString};

/// Convert a Python object to a blobfig Value
pub fn py_to_value(obj: &Bound<'_, PyAny>) -> PyResult<Value> {
    if obj.is_none() {
        return Err(PyValueError::new_err("None is not supported"));
    }

    // Bool must come before int (bool is subclass of int in Python)
    if let Ok(b) = obj.cast_exact::<PyBool>() {
        return Ok(Value::Bool(b.is_true()));
    }

    if let Ok(i) = obj.cast_exact::<PyInt>() {
        return Ok(Value::Int(i.extract()?));
    }

    if let Ok(f) = obj.cast_exact::<PyFloat>() {
        return Ok(Value::Float(f.extract()?));
    }

    if let Ok(s) = obj.cast_exact::<PyString>() {
        return Ok(Value::String(s.extract()?));
    }

    // Check for our wrapper types
    if let Ok(file) = obj.extract::<PyFile>() {
        return Ok(Value::File(File::from_bytes(&file.mimetype, file.data)));
    }

    if let Ok(arr) = obj.extract::<PyArray>() {
        return Ok(Value::Array(Array::new(arr.dtype, arr.shape, arr.data)));
    }

    // Check for raw bytes -> File
    if let Ok(b) = obj.cast_exact::<PyBytes>() {
        return Ok(Value::File(File::from_bytes(
            "application/octet-stream",
            b.as_bytes().to_vec(),
        )));
    }

    // Check for numpy arrays
    if let Ok(arr) = obj.cast::<PyArray1<f32>>() {
        let data: Vec<u8> = arr
            .readonly()
            .as_slice()?
            .iter()
            .flat_map(|f| f.to_le_bytes())
            .collect();
        return Ok(Value::Array(Array::new(
            DType::F32,
            vec![arr.len() as u64],
            data,
        )));
    }
    if let Ok(arr) = obj.cast::<PyArray1<f64>>() {
        let data: Vec<u8> = arr
            .readonly()
            .as_slice()?
            .iter()
            .flat_map(|f| f.to_le_bytes())
            .collect();
        return Ok(Value::Array(Array::new(
            DType::F64,
            vec![arr.len() as u64],
            data,
        )));
    }
    if let Ok(arr) = obj.cast::<PyArray1<i32>>() {
        let data: Vec<u8> = arr
            .readonly()
            .as_slice()?
            .iter()
            .flat_map(|i| i.to_le_bytes())
            .collect();
        return Ok(Value::Array(Array::new(
            DType::I32,
            vec![arr.len() as u64],
            data,
        )));
    }
    if let Ok(arr) = obj.cast::<PyArray1<i64>>() {
        let data: Vec<u8> = arr
            .readonly()
            .as_slice()?
            .iter()
            .flat_map(|i| i.to_le_bytes())
            .collect();
        return Ok(Value::Array(Array::new(
            DType::I64,
            vec![arr.len() as u64],
            data,
        )));
    }

    // Check for dict -> Object
    if let Ok(d) = obj.cast_exact::<PyDict>() {
        let mut entries = Vec::new();
        for (k, v) in d.iter() {
            let key: String = k.extract()?;
            let val = py_to_value(&v)?;
            entries.push((key, val));
        }
        return Ok(Value::Object(entries));
    }

    // Check for list -> List
    if let Ok(l) = obj.cast_exact::<PyList>() {
        let mut items = Vec::new();
        for item in l.iter() {
            items.push(py_to_value(&item)?);
        }
        return Ok(Value::List(items));
    }

    Err(PyValueError::new_err(format!(
        "Unsupported type: {}",
        obj.get_type().name()?
    )))
}

/// Convert a blobfig ValueView to a Python object
pub fn value_to_py<'py>(py: Python<'py>, view: &ValueView<'_>) -> PyResult<Bound<'py, PyAny>> {
    match view {
        ValueView::Bool(b) => Ok(PyBool::new(py, *b).to_owned().into_any()),
        ValueView::Int(i) => Ok(i.into_pyobject(py)?.into_any().unbind().into_bound(py)),
        ValueView::Float(f) => Ok(f.into_pyobject(py)?.into_any().unbind().into_bound(py)),
        ValueView::String(s) => Ok(PyString::new(py, s).into_any()),
        ValueView::Array(arr) => {
            let py_arr = PyArray {
                dtype: arr.dtype,
                shape: arr.shape.clone(),
                data: arr.data.to_vec(),
            };
            Ok(Py::new(py, py_arr)?.into_bound(py).into_any())
        }
        ValueView::File(f) => {
            let py_file = PyFile {
                mimetype: f.mimetype.to_string(),
                data: f.data.to_vec(),
            };
            Ok(Py::new(py, py_file)?.into_bound(py).into_any())
        }
        ValueView::Object(entries) => {
            let dict = PyDict::new(py);
            for (k, v) in entries.iter() {
                dict.set_item(*k, value_to_py(py, v)?)?;
            }
            Ok(dict.into_any())
        }
        ValueView::List(items) => {
            let list = PyList::empty(py);
            for item in items.iter() {
                list.append(value_to_py(py, item)?)?;
            }
            Ok(list.into_any())
        }
    }
}

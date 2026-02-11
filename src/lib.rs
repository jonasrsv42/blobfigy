//! Python bindings for blobfig

mod types;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use types::{PyArray, PyFile, py_to_value, value_to_py};

/// Parse blobfig bytes into a Python object
#[pyfunction]
fn parse<'py>(py: Python<'py>, data: &[u8]) -> PyResult<Bound<'py, PyAny>> {
    let view =
        blobfig::parse(data).map_err(|e| PyValueError::new_err(format!("Parse error: {}", e)))?;
    value_to_py(py, &view)
}

/// Serialize a Python object to blobfig bytes
#[pyfunction]
fn to_bytes<'py>(py: Python<'py>, obj: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyBytes>> {
    let value = py_to_value(obj)?;
    let bytes = blobfig::writer::to_bytes(value)
        .map_err(|e| PyValueError::new_err(format!("Write error: {}", e)))?;
    Ok(PyBytes::new(py, &bytes))
}

/// Python module for blobfig
#[pymodule]
fn _blobfigy(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyArray>()?;
    m.add_class::<PyFile>()?;
    m.add_function(wrap_pyfunction!(parse, m)?)?;
    m.add_function(wrap_pyfunction!(to_bytes, m)?)?;
    Ok(())
}

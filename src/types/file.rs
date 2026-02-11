//! Python wrapper for blobfig File

use pyo3::prelude::*;
use pyo3::types::PyBytes;

/// Python wrapper for blobfig File
#[pyclass(name = "File", frozen, from_py_object)]
#[derive(Clone)]
pub struct PyFile {
    #[pyo3(get)]
    pub mimetype: String,
    pub data: Vec<u8>,
}

#[pymethods]
impl PyFile {
    #[new]
    #[pyo3(signature = (mimetype, data))]
    fn new(mimetype: String, data: Vec<u8>) -> Self {
        PyFile { mimetype, data }
    }

    #[getter]
    fn data<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        PyBytes::new(py, &self.data)
    }

    fn __repr__(&self) -> String {
        format!(
            "File(mimetype='{}', size={})",
            self.mimetype,
            self.data.len()
        )
    }
}

//! Python wrapper for blobfig Array

use blobfig::DType;
use numpy::PyArrayMethods;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyBytes;

/// Python wrapper for blobfig Array
#[pyclass(name = "Array", frozen, from_py_object)]
#[derive(Clone)]
pub struct PyArray {
    pub dtype: DType,
    pub shape: Vec<u64>,
    pub data: Vec<u8>,
}

#[pymethods]
impl PyArray {
    #[new]
    #[pyo3(signature = (dtype, shape, data))]
    fn new(dtype: &str, shape: Vec<u64>, data: Vec<u8>) -> PyResult<Self> {
        let dtype = parse_dtype(dtype)?;
        Ok(PyArray { dtype, shape, data })
    }

    #[getter]
    fn dtype(&self) -> &str {
        dtype_to_str(self.dtype)
    }

    #[getter]
    fn shape(&self) -> Vec<u64> {
        self.shape.clone()
    }

    #[getter]
    fn data<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        PyBytes::new(py, &self.data)
    }

    /// Convert to numpy array
    fn to_numpy<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let shape: Vec<usize> = self.shape.iter().map(|&d| d as usize).collect();

        match self.dtype {
            DType::U8 => {
                let arr = numpy::PyArray::from_vec(py, self.data.clone());
                Ok(arr.reshape(shape)?.into_any())
            }
            DType::I8 => {
                let elements: Vec<i8> = self.data.iter().map(|&b| b as i8).collect();
                let arr = numpy::PyArray::from_vec(py, elements);
                Ok(arr.reshape(shape)?.into_any())
            }
            DType::U16 => {
                let elements: Vec<u16> = self
                    .data
                    .chunks_exact(2)
                    .map(|c| u16::from_le_bytes([c[0], c[1]]))
                    .collect();
                let arr = numpy::PyArray::from_vec(py, elements);
                Ok(arr.reshape(shape)?.into_any())
            }
            DType::I16 => {
                let elements: Vec<i16> = self
                    .data
                    .chunks_exact(2)
                    .map(|c| i16::from_le_bytes([c[0], c[1]]))
                    .collect();
                let arr = numpy::PyArray::from_vec(py, elements);
                Ok(arr.reshape(shape)?.into_any())
            }
            DType::U32 => {
                let elements: Vec<u32> = self
                    .data
                    .chunks_exact(4)
                    .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect();
                let arr = numpy::PyArray::from_vec(py, elements);
                Ok(arr.reshape(shape)?.into_any())
            }
            DType::I32 => {
                let elements: Vec<i32> = self
                    .data
                    .chunks_exact(4)
                    .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect();
                let arr = numpy::PyArray::from_vec(py, elements);
                Ok(arr.reshape(shape)?.into_any())
            }
            DType::U64 => {
                let elements: Vec<u64> = self
                    .data
                    .chunks_exact(8)
                    .map(|c| u64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]))
                    .collect();
                let arr = numpy::PyArray::from_vec(py, elements);
                Ok(arr.reshape(shape)?.into_any())
            }
            DType::I64 => {
                let elements: Vec<i64> = self
                    .data
                    .chunks_exact(8)
                    .map(|c| i64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]))
                    .collect();
                let arr = numpy::PyArray::from_vec(py, elements);
                Ok(arr.reshape(shape)?.into_any())
            }
            DType::F32 => {
                let elements: Vec<f32> = self
                    .data
                    .chunks_exact(4)
                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect();
                let arr = numpy::PyArray::from_vec(py, elements);
                Ok(arr.reshape(shape)?.into_any())
            }
            DType::F64 => {
                let elements: Vec<f64> = self
                    .data
                    .chunks_exact(8)
                    .map(|c| f64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]))
                    .collect();
                let arr = numpy::PyArray::from_vec(py, elements);
                Ok(arr.reshape(shape)?.into_any())
            }
        }
    }

    fn __repr__(&self) -> String {
        format!("Array(dtype='{}', shape={:?})", self.dtype(), self.shape)
    }
}

/// Parse dtype string to DType
pub fn parse_dtype(s: &str) -> PyResult<DType> {
    match s {
        "u8" => Ok(DType::U8),
        "i8" => Ok(DType::I8),
        "u16" => Ok(DType::U16),
        "i16" => Ok(DType::I16),
        "u32" => Ok(DType::U32),
        "i32" => Ok(DType::I32),
        "u64" => Ok(DType::U64),
        "i64" => Ok(DType::I64),
        "f32" => Ok(DType::F32),
        "f64" => Ok(DType::F64),
        _ => Err(PyValueError::new_err(format!("Unknown dtype: {}", s))),
    }
}

/// Convert DType to string
pub fn dtype_to_str(dtype: DType) -> &'static str {
    match dtype {
        DType::U8 => "u8",
        DType::I8 => "i8",
        DType::U16 => "u16",
        DType::I16 => "i16",
        DType::U32 => "u32",
        DType::I32 => "i32",
        DType::U64 => "u64",
        DType::I64 => "i64",
        DType::F32 => "f32",
        DType::F64 => "f64",
    }
}

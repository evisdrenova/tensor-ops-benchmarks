use serde::{Deserialize, Serialize};
use ndarray::{Array1, Array2, Axis};
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Uniform;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestCase {
    pub name: String,
    pub shape: Vec<usize>,
    pub operation: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkResult {
    pub library: String,
    pub test_case: String,
    pub mean_time_ns: f64,
    pub std_dev_ns: f64,
    pub throughput_ops_per_sec: f64,
    pub memory_usage_bytes: Option<usize>,
}

pub trait TensorBenchmark {
    type Tensor;

    fn name(&self) -> &'static str;
    fn create_random_tensor(&self, shape: &[usize]) -> Self::Tensor;
    fn create_zeros(&self, shape: &[usize]) -> Self::Tensor;
    fn create_ones(&self, shape: &[usize]) -> Self::Tensor;

    // Basic operations
    fn add(&self, a: &Self::Tensor, b: &Self::Tensor) -> Self::Tensor;
    fn multiply(&self, a: &Self::Tensor, b: &Self::Tensor) -> Self::Tensor;
    fn matmul(&self, a: &Self::Tensor, b: &Self::Tensor) -> Self::Tensor;
    fn transpose(&self, tensor: &Self::Tensor) -> Self::Tensor;
    fn sum(&self, tensor: &Self::Tensor) -> f32;
    fn mean(&self, tensor: &Self::Tensor) -> f32;
}

pub struct NdArrayBenchmark;

impl TensorBenchmark for NdArrayBenchmark {
    type Tensor = Array2<f32>;

    fn name(&self) -> &'static str {
        "ndarray"
    }

    fn create_random_tensor(&self, shape: &[usize]) -> Self::Tensor {
        match shape.len() {
            1 => {
                let arr1 = Array1::random(shape[0], Uniform::new(-1.0, 1.0));
                arr1.insert_axis(Axis(1))
            }
            2 => Array2::random((shape[0], shape[1]), Uniform::new(-1.0, 1.0)),
            _ => panic!("Unsupported shape length: {}", shape.len()),
        }
    }

    fn create_zeros(&self, shape: &[usize]) -> Self::Tensor {
        match shape.len() {
            1 => Array2::zeros((shape[0], 1)),
            2 => Array2::zeros((shape[0], shape[1])),
            _ => panic!("Unsupported shape length: {}", shape.len()),
        }
    }

    fn create_ones(&self, shape: &[usize]) -> Self::Tensor {
        match shape.len() {
            1 => Array2::ones((shape[0], 1)),
            2 => Array2::ones((shape[0], shape[1])),
            _ => panic!("Unsupported shape length: {}", shape.len()),
        }
    }

    fn add(&self, a: &Self::Tensor, b: &Self::Tensor) -> Self::Tensor {
        a + b
    }

    fn multiply(&self, a: &Self::Tensor, b: &Self::Tensor) -> Self::Tensor {
        a * b
    }

    fn matmul(&self, a: &Self::Tensor, b: &Self::Tensor) -> Self::Tensor {
        a.dot(b)
    }

    fn transpose(&self, tensor: &Self::Tensor) -> Self::Tensor {
        tensor.t().to_owned()
    }

    fn sum(&self, tensor: &Self::Tensor) -> f32 {
        tensor.sum()
    }

    fn mean(&self, tensor: &Self::Tensor) -> f32 {
        tensor.mean().unwrap()
    }
}

// Helper struct for 1D operations
pub struct NdArrayBenchmark1D;

impl NdArrayBenchmark1D {
    pub fn create_random_vector(&self, size: usize) -> Array1<f32> {
        Array1::random(size, Uniform::new(-1.0, 1.0))
    }

    pub fn vector_add(&self, a: &Array1<f32>, b: &Array1<f32>) -> Array1<f32> {
        a + b
    }

    pub fn vector_multiply(&self, a: &Array1<f32>, b: &Array1<f32>) -> Array1<f32> {
        a * b
    }

    pub fn vector_dot(&self, a: &Array1<f32>, b: &Array1<f32>) -> f32 {
        a.dot(b)
    }
}

// Test case generators
pub fn generate_test_cases() -> Vec<TestCase> {
    let mut cases = Vec::new();

    // Small tensors
    for size in [10, 100, 500] {
        cases.push(TestCase {
            name: format!("vector_ops_{}", size),
            shape: vec![size],
            operation: "vector_ops".to_string(),
        });
    }

    // Medium 2D tensors
    for size in [32, 64, 128, 256, 512] {
        cases.push(TestCase {
            name: format!("matrix_ops_{}x{}", size, size),
            shape: vec![size, size],
            operation: "matrix_ops".to_string(),
        });
    }

    // Large tensors
    for size in [1024, 2048] {
        cases.push(TestCase {
            name: format!("large_matrix_{}x{}", size, size),
            shape: vec![size, size],
            operation: "matrix_ops".to_string(),
        });
    }

    // Batch operations
    for (batch, dim) in [(8, 256), (32, 128), (64, 64)] {
        cases.push(TestCase {
            name: format!("batch_ops_{}x{}", batch, dim),
            shape: vec![batch, dim],
            operation: "batch_ops".to_string(),
        });
    }

    cases
}
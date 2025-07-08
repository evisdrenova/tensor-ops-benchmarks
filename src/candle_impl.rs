// src/candle_impl.rs
use crate::TensorBenchmark;
use candle_core::{Device, Result, Tensor};

pub struct CandleBenchmark {
    device: Device,
}

impl CandleBenchmark {
    pub fn new() -> Result<Self> {
        Ok(Self {
            device: Device::Cpu,
        })
    }
}

impl TensorBenchmark for CandleBenchmark {
    type Tensor = Tensor;

    fn name(&self) -> &'static str {
        "candle"
    }

    fn create_random_tensor(&self, shape: &[usize]) -> Self::Tensor {
        match shape.len() {
            1 => Tensor::randn(0f32, 1f32, (shape[0],), &self.device).unwrap(),
            2 => Tensor::randn(0f32, 1f32, (shape[0], shape[1]), &self.device).unwrap(),
            _ => panic!("Unsupported shape length: {}", shape.len()),
        }
    }

    fn create_zeros(&self, shape: &[usize]) -> Self::Tensor {
        match shape.len() {
            1 => Tensor::zeros((shape[0],), candle_core::DType::F32, &self.device).unwrap(),
            2 => {
                Tensor::zeros((shape[0], shape[1]), candle_core::DType::F32, &self.device).unwrap()
            }
            _ => panic!("Unsupported shape length: {}", shape.len()),
        }
    }

    fn create_ones(&self, shape: &[usize]) -> Self::Tensor {
        match shape.len() {
            1 => Tensor::ones((shape[0],), candle_core::DType::F32, &self.device).unwrap(),
            2 => Tensor::ones((shape[0], shape[1]), candle_core::DType::F32, &self.device).unwrap(),
            _ => panic!("Unsupported shape length: {}", shape.len()),
        }
    }

    fn add(&self, a: &Self::Tensor, b: &Self::Tensor) -> Self::Tensor {
        a.add(b).unwrap()
    }

    fn multiply(&self, a: &Self::Tensor, b: &Self::Tensor) -> Self::Tensor {
        a.mul(b).unwrap()
    }

    fn matmul(&self, a: &Self::Tensor, b: &Self::Tensor) -> Self::Tensor {
        a.matmul(b).unwrap()
    }

    fn transpose(&self, tensor: &Self::Tensor) -> Self::Tensor {
        // For 2D tensors, transpose the last two dimensions
        if tensor.dims().len() == 2 {
            tensor.t().unwrap()
        } else {
            tensor.transpose(0, 1).unwrap()
        }
    }

    fn sum(&self, tensor: &Self::Tensor) -> f32 {
        let sum_tensor = tensor.sum_all().unwrap();
        sum_tensor.to_scalar::<f32>().unwrap()
    }

    fn mean(&self, tensor: &Self::Tensor) -> f32 {
        let sum = self.sum(tensor);
        let numel = tensor.elem_count() as f32;
        sum / numel
    }
}

impl Default for CandleBenchmark {
    fn default() -> Self {
        Self::new().unwrap()
    }
}

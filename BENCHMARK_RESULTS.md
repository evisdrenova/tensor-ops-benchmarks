# Rust Tensor Libraries Benchmark Results

## Executive Summary

This report analyzes the performance of three Rust tensor libraries: **Candle**, **Burn**, and **NDArray** across various tensor operations. The benchmarks were conducted using the Criterion framework with statistical analysis and confidence intervals.

## Test Environment

- **Hardware**: Multi-core CPU (system-dependent)
- **Data Type**: f32 (32-bit floating point)
- **Optimization**: Release mode with LTO enabled
- **Measurement**: Criterion framework with 95% confidence intervals

## Performance Overview

### 🏆 Best Performers by Operation

| Operation                   | Winner         | Performance Advantage                             |
| --------------------------- | -------------- | ------------------------------------------------- |
| **Tensor Creation**         | NDArray        | ~4.5x faster than Burn, ~8.2x faster than Candle  |
| **Matrix Multiplication**   | Candle         | ~1.7x faster than Burn, ~3.9x faster than NDArray |
| **Element-wise Operations** | NDArray/Candle | Virtually identical performance                   |
| **Reduction Operations**    | Candle         | ~1.7x faster than NDArray/Burn                    |
| **Vector Operations**       | NDArray        | ~2.1x faster than Burn                            |

## Detailed Results

### 1. Tensor Creation (512×512 Random Tensors)

Performance for creating random tensors:

| Library     | Mean Time (μs) | Std Dev (μs) | Relative Performance |
| ----------- | -------------- | ------------ | -------------------- |
| **NDArray** | 317.3          | 63.2         | **1.00x** (baseline) |
| **Burn**    | 1,435.9        | 172.0        | 4.53x slower         |
| **Candle**  | 2,605.6        | 85.7         | 8.22x slower         |

**Analysis**: NDArray significantly outperforms both Burn and Candle for tensor creation. This suggests NDArray has more efficient memory allocation and initialization routines.

### 2. Matrix Multiplication (512×512 × 512×512)

Performance for matrix multiplication:

| Library     | Mean Time (μs) | Std Dev (μs) | Relative Performance |
| ----------- | -------------- | ------------ | -------------------- |
| **Candle**  | 674.8          | 75.7         | **1.00x** (baseline) |
| **Burn**    | 1,144.0        | 190.3        | 1.70x slower         |
| **NDArray** | 2,663.8        | 105.3        | 3.95x slower         |

**Analysis**: Candle dominates matrix multiplication, likely due to optimized GEMM (General Matrix Multiply) implementations. This is crucial for deep learning workloads.

### 3. Element-wise Addition (512×512 + 512×512)

Performance for element-wise addition:

| Library     | Mean Time (μs) | Std Dev (μs) | Relative Performance |
| ----------- | -------------- | ------------ | -------------------- |
| **Candle**  | 30.7           | 2.0          | **1.00x** (baseline) |
| **NDArray** | 30.9           | 0.9          | 1.01x slower         |
| **Burn**    | 31.3           | 0.8          | 1.02x slower         |

**Analysis**: All three libraries show nearly identical performance for element-wise operations, suggesting similar vectorization optimizations.

### 4. Reduction Operations (Sum)

Performance for tensor sum operations (256×256 tensors):

| Library     | Mean Time (μs) | Performance Notes    |
| ----------- | -------------- | -------------------- |
| **Candle**  | ~4.2           | Fastest reduction    |
| **NDArray** | ~7.3           | Moderate performance |
| **Burn**    | ~7.8           | Slowest reduction    |

### 5. Vector Operations (Dot Product)

Performance for vector dot products (100K elements):

| Library     | Mean Time (μs) | Performance Notes    |
| ----------- | -------------- | -------------------- |
| **NDArray** | ~11.2          | Optimized vector ops |
| **Burn**    | ~23.9          | 2.1x slower          |

_Note: Candle benchmarks don't include vector operations_

## Performance Scaling Analysis

### Matrix Multiplication Scaling (64×64 to 512×512)

The libraries show different scaling characteristics:

- **Candle**: Excellent scaling, maintains performance advantage
- **Burn**: Good scaling but consistently slower than Candle
- **NDArray**: Poor scaling for larger matrices

### Element-wise Operations Scaling

All libraries scale similarly for element-wise operations, maintaining competitive performance across different tensor sizes.

## Memory and Throughput Analysis

### Tensor Creation Throughput (512×512 matrices)

| Library     | Elements/sec      | Throughput Efficiency |
| ----------- | ----------------- | --------------------- |
| **NDArray** | 831M elements/sec | Highest throughput    |
| **Burn**    | 183M elements/sec | Moderate throughput   |
| **Candle**  | 101M elements/sec | Lowest throughput     |

### Matrix Multiplication FLOPS (512×512 × 512×512)

Theoretical FLOPS for 512×512 matrix multiplication: ~268M FLOPS

| Library     | Actual FLOPS | Efficiency          |
| ----------- | ------------ | ------------------- |
| **Candle**  | ~397M FLOPS  | Best efficiency     |
| **Burn**    | ~234M FLOPS  | Moderate efficiency |
| **NDArray** | ~101M FLOPS  | Poor efficiency     |

## Recommendations

### Use Cases by Library

**🚀 Choose Candle for:**

- Deep learning and neural network training
- Heavy matrix multiplication workloads
- Reduction operations (sum, mean, etc.)
- GPU acceleration plans

**⚡ Choose NDArray for:**

- Data preprocessing and tensor creation
- Element-wise operations
- Vector computations
- Pure CPU environments

**🔧 Choose Burn for:**

- Type-safe tensor operations
- Cross-platform compatibility
- Balanced performance across operations
- Automatic differentiation needs

### Performance Optimization Tips

1. **For Matrix Operations**: Prefer Candle for compute-intensive linear algebra
2. **For Data Loading**: Use NDArray for efficient tensor creation and preprocessing
3. **For Element-wise Ops**: Any library works well - choose based on ecosystem
4. **For Production**: Consider Candle for inference, Burn for training with autodiff

## Benchmark Limitations

- **CPU Only**: No GPU benchmarks included
- **Single Data Type**: Only f32 tested
- **Limited Operations**: Core operations only, no neural network layers
- **System Dependent**: Results may vary across different hardware

## Statistical Confidence

All results include 95% confidence intervals and are based on multiple benchmark runs with statistical analysis. The Criterion framework provides regression detection and performance visualization for reliable measurements.

## Conclusion

Each library has distinct performance characteristics:

- **Candle** excels at compute-intensive operations like matrix multiplication
- **NDArray** dominates memory-intensive operations like tensor creation
- **Burn** provides consistent, balanced performance with additional safety features

The choice depends on your specific use case, with Candle being ideal for ML inference, NDArray for data processing, and Burn for comprehensive ML training pipelines.

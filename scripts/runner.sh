cat > benchmark_runner.sh << 'EOF'
#!/bin/bash
# benchmark_runner.sh - Run all benchmarks and organize results

echo "Running comprehensive tensor benchmarks..."

# Create results directory with timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RESULTS_DIR="results/run_$TIMESTAMP"
mkdir -p "$RESULTS_DIR"

echo "Results will be saved to: $RESULTS_DIR"

# Run benchmarks with different configurations
echo "Running benchmarks in release mode..."
cargo bench --bench tensor_ops -- --output-format html

# Copy criterion results
cp -r target/criterion "$RESULTS_DIR/"

# Generate summary report
echo "Generating summary report..."
python3 << 'PYTHON'
import json
import os
from pathlib import Path

results_dir = os.environ.get('RESULTS_DIR', 'results/latest')
criterion_dir = Path(f"{results_dir}/criterion")

if criterion_dir.exists():
    print(f"Benchmark results available at: {
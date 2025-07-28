import torch
import time
import torch_xla
import torch_xla.core.xla_model as xm

# --- Benchmark Configuration ---
BENCHMARK_ITERATIONS = 100

def run_matmul_on_tpu():
    # Set the device to TPU
    device = xm.xla_device()
    print(f"Using device: {device}")

    # Define matrix dimensions (conceptual for an LLM layer)
    # batch_size: number of input sequences processed in parallel
    # hidden_size: dimension of the hidden states (e.g., 4096, 8192)
    # intermediate_size: typically 4x hidden_size in Transformer FFNs
    batch_size = 128 # Set to match CPU for fair comparison. Can be lowered if OOM.
    hidden_size = 4096
    intermediate_size = 16384 # 4 * hidden_size for FFN

    print(f"Simulating MatMul for LLM Feed-Forward Network:")
    print(f"  Batch size: {batch_size}")
    print(f"  Hidden size: {hidden_size}")
    print(f"  Intermediate size: {intermediate_size}")

    # Create random input tensor and weight matrix on the TPU device
    input_tensor = torch.randn(batch_size, hidden_size, dtype=torch.float32).to(device)
    weight_matrix = torch.randn(hidden_size, intermediate_size, dtype=torch.float32).to(device)

    print(f"\nInput tensor shape: {input_tensor.shape} (dtype: {input_tensor.dtype})")
    print(f"Weight matrix shape: {weight_matrix.shape} (dtype: {weight_matrix.dtype})")

    # --- Benchmark Phase ---
    print(f"Performing {BENCHMARK_ITERATIONS} benchmark iterations...")
    start_time = time.perf_counter()
    for _ in range(BENCHMARK_ITERATIONS):
        output_tensor = torch.matmul(input_tensor, weight_matrix)
    xm.mark_step() # Ensure all benchmark operations are complete before stopping timer
    end_time = time.perf_counter()
    total_time_ms = (end_time - start_time) * 1000

    print(f"\n--- Results ---")
    print(f"Total time for {BENCHMARK_ITERATIONS} iterations: {total_time_ms:.4f} ms")
    print(f"Average time per iteration: {total_time_ms / BENCHMARK_ITERATIONS:.4f} ms")
    # Move output tensor to CPU for further processing/inspection if needed
    output_tensor = output_tensor.cpu()

    # Clean up (optional)
    del input_tensor, weight_matrix, output_tensor

if __name__ == "__main__":
    try:
        print("--- Matrix Multiplication Benchmark (TPU) ---")
        run_matmul_on_tpu()
    except Exception as e:
        print(f"An error occurred: {e}")
        print("Please ensure you have PyTorch and Torch_XLA installed to run this code and you are running in a TPU environment.")

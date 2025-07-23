import torch
import time

def run_matmul_on_cpu():
    # Set the device explicitly to CPU
    device = torch.device("cpu")
    print(f"Using device: {device}")

    # Define matrix dimensions (conceptual for an LLM layer)
    # batch_size: number of input sequences processed in parallel
    # hidden_size: dimension of the hidden states (e.g., 4096, 8192)
    # intermediate_size: typically 4x hidden_size in Transformer FFNs
    batch_size = 128
    hidden_size = 4096
    intermediate_size = 16384 # 4 * hidden_size for FFN

    print(f"Simulating MatMul for LLM Feed-Forward Network:")
    print(f"  Batch size: {batch_size}")
    print(f"  Input features (hidden_size): {hidden_size}")
    print(f"  Output features (intermediate_size): {intermediate_size}")

    # Create random input tensor and weight matrix on the CPU device
    # Using float32 (standard float) for CPU computation
    input_tensor = torch.randn(batch_size, hidden_size, dtype=torch.float32).to(device)
    weight_matrix = torch.randn(hidden_size, intermediate_size, dtype=torch.float32).to(device)

    print(f"\nInput tensor shape: {input_tensor.shape} (dtype: {input_tensor.dtype})")
    print(f"Weight matrix shape: {weight_matrix.shape} (dtype: {weight_matrix.dtype})")

    # Perform the matrix multiplication
    print("\nPerforming matrix multiplication (input @ weight_matrix)...")
    start_time = time.perf_counter()
    output_tensor = torch.matmul(input_tensor, weight_matrix)
    # No xm.mark_step() needed for CPU as operations are typically eagerly executed

    end_time = time.perf_counter()

    print(f"Output tensor shape: {output_tensor.shape}")
    print(f"Computation time: {(end_time - start_time) * 1000:.4f} ms")

    # Clean up (optional)
    del input_tensor, weight_matrix, output_tensor

if __name__ == "__main__":
    try:
        print("--- Matrix Multiplication Benchmark (CPU) ---")
        run_matmul_on_cpu()
    except Exception as e:
        print(f"An error occurred: {e}")
        print("Please ensure you have PyTorch installed to run this code.")

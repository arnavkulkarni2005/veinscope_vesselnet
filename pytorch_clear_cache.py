import torch

# Clear the PyTorch cache
torch.cuda.empty_cache()

# Optionally, reset the allocated memory
torch.cuda.reset_max_memory_allocated()
torch.cuda.reset_max_memory_cached()

print("PyTorch cache and allocated memory cleared.")

# Display memory usage
allocated_memory = torch.cuda.memory_allocated() / 1e6  # Convert to MB
cached_memory = torch.cuda.memory_reserved() / 1e6  # Convert to MB
print(f"Allocated memory: {allocated_memory:.2f} MB")
print(f"Cached memory: {cached_memory:.2f} MB")

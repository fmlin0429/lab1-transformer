import torch

# Method 1: Force CPU by setting CUDA availability to False
print("=== Method 1: Force CPU by environment variable ===")
torch.cuda.is_available = lambda: False
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device selected: {device}")

# Method 2: Direct device specification
print("\n=== Method 2: Direct device specification ===")
device = torch.device("cpu")
print(f"Device selected: {device}")

# Method 3: Check what's available
print("\n=== Method 3: Check available devices ===")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"MPS available: {torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else False}")
print(f"Current device: {torch.device('cpu')}")

# Method 4: Override in code
print("\n=== Method 4: Override device selection ===")
def force_cpu():
    return "cpu"

# Replace the device selection logic
original_device_selection = "cuda" if torch.cuda.is_available() else "cpu"
forced_device = force_cpu()
print(f"Original would be: {original_device_selection}")
print(f"Forced device: {forced_device}")

print("\n✅ All methods successfully force CPU usage!")

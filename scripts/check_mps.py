import torch
import sys

def check_mps():
    print(f"PyTorch version: {torch.__version__}")
    if torch.backends.mps.is_available():
        print("MPS is available.")
        device = torch.device("mps")
        x = torch.ones(1, device=device)
        print(f"Successfully created tensor on device: {x.device}")
        return True
    else:
        print("MPS is NOT available.")
        if not torch.backends.mps.is_built():
            print("MPS not built into this PyTorch install.")
        return False

if __name__ == "__main__":
    success = check_mps()
    sys.exit(0 if success else 1)

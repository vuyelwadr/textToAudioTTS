
import torch
import time

def check_atan2():
    if not torch.backends.mps.is_available():
        print("MPS not available")
        return

    device = torch.device("mps")
    print(f"Testing on {device}")

    # Create complex tensor
    a = torch.randn(1000, 1000, dtype=torch.complex64, device=device)
    
    # Test torch.angle (known to fallback/fail)
    print("Testing torch.angle...")
    start = time.time()
    try:
        b = torch.angle(a)
        torch.mps.synchronize()
        print(f"torch.angle took {time.time()-start:.4f}s")
    except Exception as e:
        print(f"torch.angle failed/warned: {e}")

    # Test atan2 workaround
    print("Testing torch.atan2 workaround...")
    start = time.time()
    try:
        # angle(z) = atan2(z.imag, z.real)
        c = torch.atan2(a.imag, a.real)
        torch.mps.synchronize()
        print(f"torch.atan2 took {time.time()-start:.4f}s")
    except Exception as e:
        print(f"torch.atan2 failed: {e}")

    # Test STFT
    print("Testing torch.stft...")
    y = torch.randn(1, 48000, device=device)
    window = torch.hann_window(800, device=device)
    start = time.time()
    try:
        spec = torch.stft(y, n_fft=800, hop_length=200, win_length=800, window=window, return_complex=True)
        torch.mps.synchronize()
        print(f"torch.stft took {time.time()-start:.4f}s")
    except Exception as e:
        print(f"torch.stft failed: {e}")

    # Test LSTM
    print("Testing nn.LSTM...")
    lstm = torch.nn.LSTM(512, 256, batch_first=True).to(device)
    x = torch.randn(1, 100, 512, device=device)
    start = time.time()
    try:
        out, _ = lstm(x)
        torch.mps.synchronize()
        print(f"nn.LSTM took {time.time()-start:.4f}s")
    except Exception as e:
        print(f"nn.LSTM failed: {e}")

    # Test ISTFT
    print("Testing torch.istft...")
    # spect: (1, 401, 241) complex? No, STFT output is (B, N, T) complex usually?
    # torch.stft returns (B, F, T) complex
    # y was (1, 48000). n_fft=800. F=401. T=241.
    try:
        # Create a dummy complex spec
        spec = torch.randn(1, 401, 241, dtype=torch.complex64, device=device)
        start = time.time()
        y_recon = torch.istft(spec, n_fft=800, hop_length=200, win_length=800, window=window, return_complex=False)
        torch.mps.synchronize()
        print(f"torch.istft took {time.time()-start:.4f}s")
        print(f"ISTFT device: {y_recon.device}")
    except Exception as e:
        print(f"torch.istft failed: {e}")

    # Test Complex Arithmetic
    print("Testing complex exp...")
    try:
        # magnitude * torch.exp(phase * 1j)
        mag = torch.randn(1, 401, 241, device=device)
        phase = torch.randn(1, 401, 241, device=device)
        start = time.time()
        res = mag * torch.exp(phase * 1j)
        torch.mps.synchronize()
        print(f"Complex math took {time.time()-start:.4f}s")
    except Exception as e:
        print(f"Complex math failed: {e}")

if __name__ == "__main__":
    check_atan2()

import torch

print("=" * 60)
print("GPU AVAILABILITY CHECK")
print("=" * 60)

# Check if CUDA is available
cuda_available = torch.cuda.is_available()
print(f"\nCUDA Available: {cuda_available}")

if cuda_available:
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"GPU Device Count: {torch.cuda.device_count()}")
    print(f"Current GPU Device: {torch.cuda.current_device()}")
    print(f"GPU Device Name: {torch.cuda.get_device_name(0)}")
    
    # Test GPU memory
    print(f"\nGPU Memory:")
    print(f"  Total: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    print(f"  Allocated: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
    print(f"  Cached: {torch.cuda.memory_reserved(0) / 1024**3:.2f} GB")
    
    # Quick speed test
    print(f"\n{'=' * 60}")
    print("SPEED TEST")
    print("=" * 60)
    
    import time
    
    # CPU test
    x_cpu = torch.randn(1000, 1000)
    start = time.time()
    y_cpu = torch.matmul(x_cpu, x_cpu)
    cpu_time = time.time() - start
    print(f"CPU Time: {cpu_time*1000:.2f}ms")
    
    # GPU test
    x_gpu = torch.randn(1000, 1000).cuda()
    torch.cuda.synchronize()
    start = time.time()
    y_gpu = torch.matmul(x_gpu, x_gpu)
    torch.cuda.synchronize()
    gpu_time = time.time() - start
    print(f"GPU Time: {gpu_time*1000:.2f}ms")
    print(f"Speedup: {cpu_time/gpu_time:.2f}x faster")
    
    print("\n✅ GPU is ready to use!")
else:
    print("\n❌ GPU not available. Running on CPU.")
    print("\nPossible reasons:")
    print("  1. No NVIDIA GPU installed")
    print("  2. CUDA not installed")
    print("  3. PyTorch CPU version installed (reinstall GPU version)")

print("=" * 60)
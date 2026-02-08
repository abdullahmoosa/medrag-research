#!/usr/bin/env python3
"""Test GPU availability and PyTorch CUDA support."""

import sys
import torch

def test_gpu_support():
    """Test GPU support and print detailed information."""
    print("=" * 60)
    print("🔍 GPU Support Test")
    print("=" * 60)
    
    # PyTorch version
    print(f"PyTorch version: {torch.__version__}")
    
    # CUDA availability
    cuda_available = torch.cuda.is_available()
    print(f"CUDA available: {cuda_available}")
    
    if cuda_available:
        # GPU count
        gpu_count = torch.cuda.device_count()
        print(f"GPU count: {gpu_count}")
        
        # Current GPU
        current_gpu = torch.cuda.current_device()
        print(f"Current GPU: {current_gpu}")
        
        # GPU names
        for i in range(gpu_count):
            gpu_name = torch.cuda.get_device_name(i)
            memory_total = torch.cuda.get_device_properties(i).total_memory / 1024**3
            print(f"GPU {i}: {gpu_name} ({memory_total:.1f} GB)")
        
        # Test tensor creation
        try:
            test_tensor = torch.randn(100, 100).cuda()
            print(f"✅ GPU tensor creation successful")
            print(f"Test tensor device: {test_tensor.device}")
            del test_tensor  # Free memory
        except Exception as e:
            print(f"❌ GPU tensor creation failed: {e}")
    else:
        print("❌ No GPU support available")
        print("Possible causes:")
        print("- CUDA not installed")
        print("- PyTorch CPU-only version")
        print("- GPU drivers not installed")
    
    print("=" * 60)

if __name__ == "__main__":
    test_gpu_support()

import torch
import time
import numpy as np

def measure_model_efficiency(model, input_shape=(1, 1, 128, 128, 128), device='cuda', num_runs=100):
    """
    Simple function to measure model efficiency
    
    Args:
        model: Your PyTorch model
        input_shape: Input tensor shape (default: 3D medical image)
        device: 'cuda' or 'cpu'
        num_runs: Number of runs for timing
    
    Returns:
        dict: Efficiency metrics
    """
    # Setup
    device = device if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    model.eval()
    
    print(f"\n🔍 Measuring efficiency for model on {device}...")
    print(f"  Input shape: {input_shape}")
    
    # 1. Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model_size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024 / 1024
    
    print(f"\n📊 Model Size:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Model size: {model_size_mb:.2f} MB")
    
    # 2. Measure FLOPs (if thop is available)
    try:
        from thop import profile, clever_format
        dummy_input = torch.randn(input_shape).to(device)
        with torch.no_grad():
            flops, params = profile(model, inputs=(dummy_input,), verbose=False)
        flops_str, params_str = clever_format([flops, params], "%.3f")
        print(f"\n🧮 Computational Complexity:")
        print(f"  FLOPs: {flops_str}")
        print(f"  MACs: {clever_format([flops/2], '%.3f')[0]}")
    except ImportError:
        print("\n🧮 Computational Complexity:")
        print("  Install thop for FLOPs calculation: pip install thop")
        flops = None
    
    # 3. Measure inference time
    dummy_input = torch.randn(input_shape).to(device)
    
    # Warm up
    print(f"\n⏱️  Measuring inference time...")
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy_input)
    
    # Actual timing
    times = []
    with torch.no_grad():
        for _ in range(num_runs):
            if device == 'cuda':
                torch.cuda.synchronize()
            
            start = time.perf_counter()
            _ = model(dummy_input)
            
            if device == 'cuda':
                torch.cuda.synchronize()
            
            times.append((time.perf_counter() - start) * 1000)  # Convert to ms
    
    mean_time = np.mean(times)
    std_time = np.std(times)
    
    print(f"  Average time: {mean_time:.2f} ± {std_time:.2f} ms")
    print(f"  FPS: {1000/mean_time:.2f}")
    
    # 4. Measure memory usage
    if device == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        with torch.no_grad():
            _ = model(dummy_input)
        
        memory_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
        print(f"\n💾 GPU Memory Usage:")
        print(f"  Peak memory: {memory_mb:.2f} MB")
    
    # Summary
    print(f"\n{'='*50}")
    print(f"✅ SUMMARY: {total_params/1e6:.2f}M params | {mean_time:.1f}ms | {1000/mean_time:.1f} FPS")
    if device == 'cuda':
        print(f"            Memory: {memory_mb:.1f}MB | Device: {torch.cuda.get_device_name()}")
    print(f"{'='*50}\n")
    
    # Return results
    results = {
        'total_params': total_params,
        'model_size_mb': model_size_mb,
        'mean_time_ms': mean_time,
        'fps': 1000/mean_time,
    }
    
    if device == 'cuda':
        results['gpu_memory_mb'] = memory_mb
        
    if flops is not None:
        results['flops'] = flops
    
    return results


# Even simpler one-liner version
def quick_check(model, input_shape=(1, 1, 128, 128, 128)):
    """Ultra-simple efficiency check - just call this!"""
    return measure_model_efficiency(model, input_shape)


# Usage example:
if __name__ == "__main__":
    # Create a dummy model for testing
    import torch.nn as nn
    
    class SimpleUNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv3d(1, 64, 3, padding=1)
            self.conv2 = nn.Conv3d(64, 64, 3, padding=1)
            self.conv3 = nn.Conv3d(64, 1, 1)
            
        def forward(self, x):
            x = torch.relu(self.conv1(x))
            x = torch.relu(self.conv2(x))
            return self.conv3(x)
    
    # Test the monitor
    model = SimpleUNet()
    
    # Just one line to get all efficiency metrics!
    results = quick_check(model)
    
    # Access specific metrics if needed
    print(f"\nYou can access specific metrics:")
    print(f"Model has {results['total_params']/1e6:.2f}M parameters")
    print(f"Inference takes {results['mean_time_ms']:.2f}ms on average")
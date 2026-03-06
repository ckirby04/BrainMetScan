"""
GPU Benchmark: Compare RTX 5060 Ti vs RTX 3070 Ti
Runs identical tests to model/gpu_benchmark.json for direct comparison.
"""

import torch
import torch.nn as nn
import time
import json
import sys
import os
import gc

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

def warmup_gpu():
    """Warmup GPU with small operations."""
    x = torch.randn(1024, 1024, device='cuda')
    for _ in range(20):
        x = x @ x
    torch.cuda.synchronize()
    del x
    torch.cuda.empty_cache()

def benchmark_matmul(size=4096, n_iters=100):
    """Matrix multiplication benchmark at FP32 and FP16."""
    results = {}
    for dtype, label in [(torch.float32, 'FP32'), (torch.float16, 'FP16')]:
        a = torch.randn(size, size, device='cuda', dtype=dtype)
        b = torch.randn(size, size, device='cuda', dtype=dtype)

        # Warmup
        for _ in range(10):
            c = a @ b
        torch.cuda.synchronize()

        # Benchmark
        start = time.perf_counter()
        for _ in range(n_iters):
            c = a @ b
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        time_per_iter = elapsed / n_iters * 1000  # ms
        flops = 2 * size**3  # multiply-add
        tflops = flops * n_iters / elapsed / 1e12

        results[label] = {
            'tflops': round(tflops, 2),
            'time_per_iter_ms': round(time_per_iter, 2)
        }
        print(f"  MatMul {size}x{size} {label}: {tflops:.2f} TFLOPS ({time_per_iter:.2f} ms/iter)")

        del a, b, c
        torch.cuda.empty_cache()

    return results

def benchmark_conv3d():
    """3D convolution benchmark (core op for segmentation)."""
    conv = nn.Conv3d(128, 128, 3, padding=1).cuda()
    x = torch.randn(1, 128, 32, 32, 32, device='cuda')

    # Warmup
    for _ in range(10):
        y = conv(x)
    torch.cuda.synchronize()

    n_iters = 100
    start = time.perf_counter()
    for _ in range(n_iters):
        y = conv(x)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    time_per_iter = elapsed / n_iters * 1000
    iters_per_sec = n_iters / elapsed

    print(f"  Conv3D 128ch: {time_per_iter:.2f} ms/iter ({iters_per_sec:.1f} iter/s)")

    del conv, x, y
    torch.cuda.empty_cache()

    return {
        'time_per_iter_ms': round(time_per_iter, 2),
        'iters_per_sec': round(iters_per_sec, 1)
    }

def benchmark_unet_custom():
    """Benchmark our LightweightUNet3D at various patch sizes."""
    from src.segmentation.unet import LightweightUNet3D

    results = {}
    for patch_size in [64, 96, 128, 192]:
        label = f"{patch_size}^3"
        torch.cuda.empty_cache()
        gc.collect()
        torch.cuda.reset_peak_memory_stats()

        try:
            model = LightweightUNet3D(in_channels=4, out_channels=1).cuda()
            model.train()
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
            x = torch.randn(1, 4, patch_size, patch_size, patch_size, device='cuda')
            target = torch.randint(0, 2, (1, 1, patch_size, patch_size, patch_size), device='cuda', dtype=torch.float32)

            # Warmup
            for _ in range(3):
                optimizer.zero_grad()
                out = model(x)
                loss = nn.functional.binary_cross_entropy_with_logits(out, target)
                loss.backward()
                optimizer.step()
            torch.cuda.synchronize()

            # Benchmark (full train step: forward + backward + optimizer)
            n_iters = 20
            start = time.perf_counter()
            for _ in range(n_iters):
                optimizer.zero_grad()
                out = model(x)
                loss = nn.functional.binary_cross_entropy_with_logits(out, target)
                loss.backward()
                optimizer.step()
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start

            time_per_step = elapsed / n_iters * 1000
            peak_vram = torch.cuda.max_memory_allocated() / 1024**3

            results[label] = {
                'time_per_step_ms': round(time_per_step, 1),
                'peak_vram_gb': round(peak_vram, 2)
            }
            print(f"  Custom UNet {label}: {time_per_step:.1f} ms/step, {peak_vram:.2f} GB VRAM")

            del model, optimizer, x, target, out, loss

        except RuntimeError as e:
            if 'out of memory' in str(e).lower() or 'cudnn' in str(e).lower():
                results[label] = {'error': 'OOM', 'peak_vram_gb': 'N/A'}
                print(f"  Custom UNet {label}: OOM ({type(e).__name__})")
            else:
                raise

        torch.cuda.empty_cache()
        gc.collect()

    return results

def benchmark_unet_nnunet():
    """Benchmark nnU-Net PlainConvUNet at various sizes."""
    try:
        from dynamic_network_architectures.architectures.unet import PlainConvUNet
    except ImportError:
        print("  nnU-Net architecture not available, skipping")
        return {}

    # nnU-Net standard config for brain mets
    results = {}
    configs = [
        ('96^3_bs2', 96, 2),
        ('128^3_bs2', 128, 2),
        ('160^3_bs2', 160, 2),
        ('192^3_bs2', 192, 2),
    ]

    for label, patch_size, batch_size in configs:
        torch.cuda.empty_cache()
        gc.collect()
        torch.cuda.reset_peak_memory_stats()

        try:
            model = PlainConvUNet(
                input_channels=4,
                n_stages=6,
                features_per_stage=[32, 64, 128, 256, 320, 320],
                conv_op=nn.Conv3d,
                kernel_sizes=[[3,3,3]]*6,
                strides=[[1,1,1]] + [[2,2,2]]*5,
                num_classes=4,
                n_conv_per_stage=[2]*6,
                n_conv_per_stage_decoder=[2]*5,
                conv_bias=True,
                norm_op=nn.InstanceNorm3d,
                nonlin=nn.LeakyReLU,
                deep_supervision=True,
            ).cuda()
            model.train()
            optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.99)

            x = torch.randn(batch_size, 4, patch_size, patch_size, patch_size, device='cuda')
            target = torch.randint(0, 4, (batch_size, patch_size, patch_size, patch_size), device='cuda')

            # Warmup
            for _ in range(3):
                optimizer.zero_grad()
                out = model(x)
                if isinstance(out, (list, tuple)):
                    out_main = out[0]
                else:
                    out_main = out
                loss = nn.functional.cross_entropy(out_main, target)
                loss.backward()
                optimizer.step()
            torch.cuda.synchronize()

            # Benchmark
            n_iters = 10
            start = time.perf_counter()
            for _ in range(n_iters):
                optimizer.zero_grad()
                out = model(x)
                if isinstance(out, (list, tuple)):
                    out_main = out[0]
                else:
                    out_main = out
                loss = nn.functional.cross_entropy(out_main, target)
                loss.backward()
                optimizer.step()
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start

            time_per_step = elapsed / n_iters * 1000
            peak_vram = torch.cuda.max_memory_allocated() / 1024**3

            results[label] = {
                'time_per_step_ms': round(time_per_step, 1),
                'peak_vram_gb': round(peak_vram, 2)
            }
            print(f"  nnU-Net {label}: {time_per_step:.1f} ms/step, {peak_vram:.2f} GB VRAM")

            del model, optimizer, x, target, out, out_main, loss

        except RuntimeError as e:
            if 'out of memory' in str(e).lower() or 'cudnn' in str(e).lower():
                results[label] = {'error': 'OOM', 'peak_vram_gb': 'N/A'}
                print(f"  nnU-Net {label}: OOM ({type(e).__name__})")
            else:
                raise

        torch.cuda.empty_cache()
        gc.collect()

    return results

def benchmark_memory_bandwidth():
    """Test GPU memory bandwidth."""
    size = 256 * 1024 * 1024  # 256M elements = 1 GB at FP32
    a = torch.randn(size, device='cuda')
    b = torch.randn(size, device='cuda')

    # Warmup
    for _ in range(5):
        c = a + b
    torch.cuda.synchronize()

    n_iters = 50
    start = time.perf_counter()
    for _ in range(n_iters):
        c = a + b
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    # Each iter reads 2 arrays + writes 1 = 3 GB transferred
    bytes_transferred = 3 * size * 4 * n_iters  # FP32 = 4 bytes
    bandwidth_gb_s = bytes_transferred / elapsed / 1e9

    print(f"  Memory bandwidth: {bandwidth_gb_s:.1f} GB/s")

    del a, b, c
    torch.cuda.empty_cache()

    return {'bandwidth_gb_s': round(bandwidth_gb_s, 1)}

def load_old_benchmark():
    """Load the RTX 3070 Ti benchmark results."""
    path = os.path.join(os.path.dirname(__file__), '..', 'model', 'gpu_benchmark.json')
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None

def print_comparison(old, new):
    """Print side-by-side comparison table."""
    old_gpu = old['gpu_info']['name']
    new_gpu = new['gpu_info']['name']

    print(f"\n{'='*72}")
    print(f"  GPU BENCHMARK COMPARISON")
    print(f"  {old_gpu} vs {new_gpu}")
    print(f"{'='*72}")

    # GPU specs
    print(f"\n{'Spec':<25} {'3070 Ti':>15} {'5060 Ti':>15} {'Change':>12}")
    print(f"{'-'*67}")
    print(f"{'VRAM (GB)':<25} {old['gpu_info']['vram_gb']:>15.1f} {new['gpu_info']['vram_gb']:>15.1f} {new['gpu_info']['vram_gb']/old['gpu_info']['vram_gb']:>11.1f}x")
    print(f"{'CUDA version':<25} {old['gpu_info']['cuda_version']:>15} {new['gpu_info']['cuda_version']:>15}")
    print(f"{'Compute capability':<25} {old['gpu_info']['compute_capability']:>15} {new['gpu_info']['compute_capability']:>15}")

    # MatMul
    print(f"\n--- Matrix Multiply (4096x4096) ---")
    print(f"{'Test':<25} {'3070 Ti':>15} {'5060 Ti':>15} {'Speedup':>12}")
    print(f"{'-'*67}")
    for dtype in ['FP32', 'FP16']:
        old_tflops = old['matmul_4096'][dtype]['tflops']
        new_tflops = new['matmul_4096'][dtype]['tflops']
        speedup = new_tflops / old_tflops
        print(f"{dtype + ' TFLOPS':<25} {old_tflops:>15.2f} {new_tflops:>15.2f} {speedup:>11.2f}x")

    # Conv3D
    print(f"\n--- Conv3D 128ch ---")
    print(f"{'Metric':<25} {'3070 Ti':>15} {'5060 Ti':>15} {'Speedup':>12}")
    print(f"{'-'*67}")
    old_conv = old['conv3d_128ch']['time_per_iter_ms']
    new_conv = new['conv3d_128ch']['time_per_iter_ms']
    print(f"{'Time (ms/iter)':<25} {old_conv:>15.2f} {new_conv:>15.2f} {old_conv/new_conv:>11.2f}x")

    # Memory bandwidth
    if 'memory_bandwidth' in new:
        print(f"\n--- Memory Bandwidth ---")
        new_bw = new['memory_bandwidth']['bandwidth_gb_s']
        print(f"{'Bandwidth (GB/s)':<25} {'N/A':>15} {new_bw:>15.1f}")

    # Custom UNet
    print(f"\n--- LightweightUNet3D (train step) ---")
    print(f"{'Patch Size':<15} {'3070 Ti ms':>12} {'5060 Ti ms':>12} {'Speedup':>10} {'3070Ti VRAM':>12} {'5060Ti VRAM':>12}")
    print(f"{'-'*73}")
    all_keys = set(list(old.get('unet_custom', {}).get('results', {}).keys()) +
                   list(new.get('unet_custom', {}).get('results', {}).keys()))
    for key in sorted(all_keys, key=lambda x: int(x.split('^')[0])):
        old_res = old.get('unet_custom', {}).get('results', {}).get(key)
        new_res = new.get('unet_custom', {}).get('results', {}).get(key)

        old_ms = f"{old_res['time_per_step_ms']:.1f}" if old_res and 'time_per_step_ms' in old_res else 'OOM/N/A'
        new_ms = f"{new_res['time_per_step_ms']:.1f}" if new_res and 'time_per_step_ms' in new_res else 'OOM/N/A'
        old_vram = f"{old_res['peak_vram_gb']:.2f}" if old_res and 'peak_vram_gb' in old_res and old_res['peak_vram_gb'] != 'N/A' else 'N/A'
        new_vram = f"{new_res['peak_vram_gb']:.2f}" if new_res and 'peak_vram_gb' in new_res and new_res['peak_vram_gb'] != 'N/A' else 'N/A'

        if old_res and new_res and 'time_per_step_ms' in old_res and 'time_per_step_ms' in new_res:
            speedup = f"{old_res['time_per_step_ms']/new_res['time_per_step_ms']:.2f}x"
        else:
            speedup = 'N/A'

        print(f"{key:<15} {old_ms:>12} {new_ms:>12} {speedup:>10} {old_vram:>12} {new_vram:>12}")

    # nnU-Net
    print(f"\n--- nnU-Net PlainConvUNet (train step) ---")
    print(f"{'Config':<15} {'3070 Ti ms':>12} {'5060 Ti ms':>12} {'Speedup':>10} {'3070Ti VRAM':>12} {'5060Ti VRAM':>12}")
    print(f"{'-'*73}")
    all_keys = set(list(old.get('unet_nnunet', {}).get('results', {}).keys()) +
                   list(new.get('unet_nnunet', {}).get('results', {}).keys()))
    for key in sorted(all_keys, key=lambda x: int(x.split('^')[0])):
        old_res = old.get('unet_nnunet', {}).get('results', {}).get(key)
        new_res = new.get('unet_nnunet', {}).get('results', {}).get(key)

        old_ms = f"{old_res['time_per_step_ms']:.1f}" if old_res and 'time_per_step_ms' in old_res else 'OOM/N/A'
        new_ms = f"{new_res['time_per_step_ms']:.1f}" if new_res and 'time_per_step_ms' in new_res else 'OOM/N/A'
        old_vram = f"{old_res['peak_vram_gb']:.2f}" if old_res and 'peak_vram_gb' in old_res and old_res['peak_vram_gb'] != 'N/A' else 'N/A'
        new_vram = f"{new_res['peak_vram_gb']:.2f}" if new_res and 'peak_vram_gb' in new_res and new_res['peak_vram_gb'] != 'N/A' else 'N/A'

        if old_res and new_res and 'time_per_step_ms' in old_res and 'time_per_step_ms' in new_res and 'error' not in old_res and 'error' not in new_res:
            speedup = f"{old_res['time_per_step_ms']/new_res['time_per_step_ms']:.2f}x"
        elif new_res and 'time_per_step_ms' in new_res and 'error' not in new_res:
            speedup = 'NEW'
        else:
            speedup = 'N/A'

        print(f"{key:<15} {old_ms:>12} {new_ms:>12} {speedup:>10} {old_vram:>12} {new_vram:>12}")

    # Note about VRAM-limited results on 3070 Ti
    print(f"\n* 3070 Ti 128^3/160^3 nnU-Net times were VRAM-limited (spilling to system RAM)")
    print(f"  Those results show the impact of insufficient VRAM, not raw GPU speed.")


def main():
    print("=" * 60)
    print("  GPU BENCHMARK - RTX 5060 Ti")
    print("=" * 60)

    # GPU info
    props = torch.cuda.get_device_properties(0)
    gpu_info = {
        'name': torch.cuda.get_device_name(0),
        'vram_gb': round(props.total_memory / 1024**3, 2),
        'compute_capability': f"{props.major}.{props.minor}",
        'sm_count': props.multi_processor_count,
        'cuda_version': torch.version.cuda,
        'pytorch_version': torch.__version__,
    }
    print(f"\nGPU: {gpu_info['name']}")
    print(f"VRAM: {gpu_info['vram_gb']} GB")
    print(f"SMs: {gpu_info['sm_count']}")
    print(f"Compute: {gpu_info['compute_capability']}")
    print(f"CUDA: {gpu_info['cuda_version']}")
    print(f"PyTorch: {gpu_info['pytorch_version']}")

    warmup_gpu()

    # Run benchmarks
    print(f"\n[1/5] Matrix Multiply (4096x4096)...")
    matmul = benchmark_matmul()

    print(f"\n[2/5] Conv3D 128ch...")
    conv3d = benchmark_conv3d()

    print(f"\n[3/5] Memory Bandwidth...")
    mem_bw = benchmark_memory_bandwidth()

    print(f"\n[4/5] LightweightUNet3D (full train step)...")
    unet_custom = benchmark_unet_custom()

    print(f"\n[5/5] nnU-Net PlainConvUNet (full train step)...")
    unet_nnunet = benchmark_unet_nnunet()

    # Save results
    new_results = {
        'gpu_info': gpu_info,
        'matmul_4096': matmul,
        'conv3d_128ch': conv3d,
        'memory_bandwidth': mem_bw,
        'unet_custom': {
            'model': 'LightweightUNet3D',
            'results': unet_custom,
        },
        'unet_nnunet': {
            'model': 'PlainConvUNet (nnU-Net)',
            'results': unet_nnunet,
        },
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    }

    out_path = os.path.join(os.path.dirname(__file__), '..', 'model', 'gpu_benchmark_5060ti.json')
    with open(out_path, 'w') as f:
        json.dump(new_results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    # Load old results and compare
    old_results = load_old_benchmark()
    if old_results:
        print_comparison(old_results, new_results)

    print("\nDone!")

if __name__ == '__main__':
    main()

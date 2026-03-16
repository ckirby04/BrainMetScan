"""
Performance benchmark suite for BrainMetScan inference pipeline.
Measures throughput, latency, memory usage on synthetic data.

Usage:
    python scripts/benchmark.py
    python scripts/benchmark.py --volume-sizes 64 128 256
    python scripts/benchmark.py --with-tta --iterations 10
"""

import argparse
import gc
import sys
import time
from pathlib import Path

import numpy as np
import torch


def create_synthetic_volume(size: int, n_channels: int = 4) -> torch.Tensor:
    """Create a random synthetic 4-channel volume."""
    return torch.randn(n_channels, size, size, size)


def measure_memory():
    """Get current GPU/CPU memory usage."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**2
        reserved = torch.cuda.memory_reserved() / 1024**2
        return {"gpu_allocated_mb": round(allocated, 1), "gpu_reserved_mb": round(reserved, 1)}
    return {"cpu_only": True}


def benchmark_model_forward(model, input_tensor, iterations=5, warmup=2):
    """Benchmark raw model forward pass."""
    device = next(model.parameters()).device
    x = input_tensor.unsqueeze(0).to(device)

    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(x)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    times = []
    with torch.no_grad():
        for _ in range(iterations):
            start = time.perf_counter()
            _ = model(x)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            times.append(time.perf_counter() - start)

    return {
        "mean_ms": round(np.mean(times) * 1000, 2),
        "std_ms": round(np.std(times) * 1000, 2),
        "min_ms": round(min(times) * 1000, 2),
        "max_ms": round(max(times) * 1000, 2),
    }


def benchmark_sliding_window(ensemble, volume, window_size=24, iterations=3):
    """Benchmark full sliding window inference."""
    times = []

    for _ in range(iterations):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        mem_before = measure_memory()
        start = time.perf_counter()

        result = ensemble.predict_volume(
            volume,
            window_size=window_size,
            overlap=0.25,
            use_tta=False,
            threshold=0.5,
        )

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        mem_after = measure_memory()

        times.append(elapsed)

    return {
        "mean_seconds": round(np.mean(times), 2),
        "std_seconds": round(np.std(times), 2),
        "min_seconds": round(min(times), 2),
        "max_seconds": round(max(times), 2),
        "memory_before": mem_before,
        "memory_after": mem_after,
        "lesions_found": result.get("lesion_count", 0),
    }


def benchmark_postprocessing(size=128, iterations=10):
    """Benchmark postprocessing pipeline."""
    from src.segmentation.postprocessing import full_postprocessing_pipeline

    prob_map = np.random.rand(size, size, size).astype(np.float32) * 0.3
    # Add some "lesion" regions
    prob_map[40:60, 40:60, 40:60] = 0.8
    prob_map[80:95, 80:95, 80:95] = 0.7

    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        _ = full_postprocessing_pipeline(prob_map, threshold=0.5)
        times.append(time.perf_counter() - start)

    return {
        "volume_size": size,
        "mean_ms": round(np.mean(times) * 1000, 2),
        "std_ms": round(np.std(times) * 1000, 2),
    }


def benchmark_lesion_extraction(size=128, iterations=10):
    """Benchmark lesion detail extraction."""
    from src.segmentation.postprocessing import extract_lesion_details

    mask = np.zeros((size, size, size), dtype=np.uint8)
    mask[40:60, 40:60, 40:60] = 1
    mask[80:95, 80:95, 80:95] = 1
    mask[10:15, 10:15, 10:15] = 1

    prob_map = np.random.rand(size, size, size).astype(np.float32)

    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        _ = extract_lesion_details(mask, prob_map, voxel_spacing=(1.0, 1.0, 1.0))
        times.append(time.perf_counter() - start)

    return {
        "volume_size": size,
        "mean_ms": round(np.mean(times) * 1000, 2),
        "std_ms": round(np.std(times) * 1000, 2),
    }


def run_benchmarks(
    volume_sizes=None,
    iterations=5,
    with_tta=False,
    with_model=False,
):
    """Run complete benchmark suite."""
    if volume_sizes is None:
        volume_sizes = [64, 128]

    results = {
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "benchmarks": {},
    }

    print("=" * 60)
    print("BrainMetScan Performance Benchmark Suite")
    print("=" * 60)
    print(f"Device: {results['device']}")
    if results["cuda_device"]:
        print(f"GPU: {results['cuda_device']}")
    print(f"Iterations: {iterations}")
    print()

    # 1. Postprocessing benchmarks
    print("--- Postprocessing Pipeline ---")
    for size in volume_sizes:
        r = benchmark_postprocessing(size=size, iterations=iterations)
        key = f"postprocessing_{size}"
        results["benchmarks"][key] = r
        print(f"  {size}^3: {r['mean_ms']:.1f} +/- {r['std_ms']:.1f} ms")

    print()

    # 2. Lesion extraction benchmarks
    print("--- Lesion Extraction ---")
    for size in volume_sizes:
        r = benchmark_lesion_extraction(size=size, iterations=iterations)
        key = f"lesion_extraction_{size}"
        results["benchmarks"][key] = r
        print(f"  {size}^3: {r['mean_ms']:.1f} +/- {r['std_ms']:.1f} ms")

    print()

    # 3. Model forward pass (if models available)
    if with_model:
        print("--- Model Forward Pass ---")
        try:
            from src.segmentation.unet import LightweightUNet3D

            device = "cuda" if torch.cuda.is_available() else "cpu"
            model = LightweightUNet3D(
                in_channels=4, out_channels=1, base_channels=16, depth=3,
                use_attention=True, use_residual=True,
            ).to(device).eval()

            for patch_size in [24, 36, 48]:
                x = torch.randn(4, patch_size, patch_size, patch_size)
                r = benchmark_model_forward(model, x, iterations=iterations)
                key = f"forward_pass_{patch_size}"
                results["benchmarks"][key] = r
                print(f"  Patch {patch_size}^3: {r['mean_ms']:.1f} +/- {r['std_ms']:.1f} ms")

            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        except Exception as e:
            print(f"  Skipped: {e}")

        print()

    # 4. Full inference pipeline (if ensemble available)
    if with_model:
        print("--- Full Sliding Window Inference ---")
        try:
            config_path = Path(__file__).parent.parent.parent / "configs" / "models.yaml"
            if config_path.exists():
                from src.segmentation.ensemble import SmartEnsemble

                device = "cuda" if torch.cuda.is_available() else "cpu"
                ensemble = SmartEnsemble.from_config(str(config_path), device=device)

                for size in volume_sizes:
                    if size > 128 and results["device"] == "cpu":
                        print(f"  {size}^3: Skipped (CPU too slow)")
                        continue

                    volume = create_synthetic_volume(size)
                    r = benchmark_sliding_window(ensemble, volume, iterations=min(iterations, 3))
                    key = f"full_inference_{size}"
                    results["benchmarks"][key] = r
                    print(f"  {size}^3: {r['mean_seconds']:.1f} +/- {r['std_seconds']:.1f} s")

                del ensemble
            else:
                print("  Skipped: configs/models.yaml not found")

        except Exception as e:
            print(f"  Skipped: {e}")

    print()
    print("=" * 60)
    print("Benchmark complete.")

    # Save results
    import json
    output_dir = Path(__file__).parent.parent.parent / "outputs"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "benchmark_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {output_path}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BrainMetScan Performance Benchmark")
    parser.add_argument("--volume-sizes", nargs="+", type=int, default=[64, 128],
                        help="Volume sizes to benchmark (default: 64 128)")
    parser.add_argument("--iterations", type=int, default=5,
                        help="Number of iterations per benchmark (default: 5)")
    parser.add_argument("--with-tta", action="store_true",
                        help="Include TTA benchmarks")
    parser.add_argument("--with-model", action="store_true",
                        help="Include model forward pass and full inference benchmarks")

    args = parser.parse_args()

    # Add project root to path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

    run_benchmarks(
        volume_sizes=args.volume_sizes,
        iterations=args.iterations,
        with_tta=args.with_tta,
        with_model=args.with_model,
    )

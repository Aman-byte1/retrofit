"""
Benchmarking suite: throughput, VRAM scaling, and latency measurements.

Records:
- Inference throughput (tokens/sec) at multiple sequence lengths
- Peak VRAM at each sequence length
- First-token latency
- Per-layer computation time
- Memory scaling curves
"""

import argparse
import gc
import json
import logging
import os
import time
from typing import Dict, Any, List

import numpy as np
import torch
import torch.nn as nn

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("benchmark")


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@torch.no_grad()
def benchmark_throughput(
    model: nn.Module,
    vocab_size: int,
    device: torch.device,
    seq_lengths: List[int] = [128, 256, 512, 1024, 2048],
    batch_size: int = 1,
    n_warmup: int = 3,
    n_measure: int = 10,
) -> List[Dict[str, Any]]:
    """Measure inference throughput at various sequence lengths."""
    model.eval()
    results = []
    
    for seq_len in seq_lengths:
        logger.info(f"  Benchmarking seq_len={seq_len}...")
        
        # Create dummy input
        x = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
        
        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        
        # Warmup
        try:
            for _ in range(n_warmup):
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    _ = model(x)
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            # Reset after warmup
            torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
            
            # Measure
            times = []
            for _ in range(n_measure):
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                start = time.perf_counter()
                
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    _ = model(x)
                
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                elapsed = time.perf_counter() - start
                times.append(elapsed)
            
            avg_time = np.mean(times)
            std_time = np.std(times)
            total_tokens = batch_size * seq_len
            
            result = {
                "seq_len": seq_len,
                "batch_size": batch_size,
                "avg_time_sec": avg_time,
                "std_time_sec": std_time,
                "min_time_sec": min(times),
                "max_time_sec": max(times),
                "tokens_per_sec": total_tokens / avg_time,
                "peak_vram_mb": torch.cuda.max_memory_allocated(0) / 1e6 if torch.cuda.is_available() else 0,
                "allocated_vram_mb": torch.cuda.memory_allocated(0) / 1e6 if torch.cuda.is_available() else 0,
                "status": "success",
            }
            
            logger.info(
                f"    seq_len={seq_len}: {result['tokens_per_sec']:.0f} tok/s | "
                f"VRAM={result['peak_vram_mb']:.0f}MB | "
                f"Time={avg_time*1000:.1f}ms"
            )
        
        except torch.cuda.OutOfMemoryError:
            logger.warning(f"    seq_len={seq_len}: OOM!")
            result = {
                "seq_len": seq_len,
                "batch_size": batch_size,
                "status": "oom",
                "tokens_per_sec": 0,
                "peak_vram_mb": 0,
            }
            torch.cuda.empty_cache()
        
        results.append(result)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    return results


@torch.no_grad()
def benchmark_generation_latency(
    model: nn.Module,
    vocab_size: int,
    device: torch.device,
    prompt_len: int = 32,
    gen_len: int = 128,
    batch_size: int = 1,
    n_runs: int = 5,
) -> Dict[str, Any]:
    """Measure autoregressive generation latency (token-by-token)."""
    model.eval()
    
    prompt = torch.randint(0, vocab_size, (batch_size, prompt_len), device=device)
    
    # Warmup
    try:
        for _ in range(2):
            tokens = prompt.clone()
            for _ in range(min(gen_len, 10)):
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    logits, _ = model(tokens)
                next_token = logits[:, -1].argmax(dim=-1, keepdim=True)
                tokens = torch.cat([tokens, next_token], dim=1)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        # Measure
        all_times = []
        first_token_times = []
        
        for run in range(n_runs):
            tokens = prompt.clone()
            token_times = []
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            for t in range(gen_len):
                start = time.perf_counter()
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    logits, _ = model(tokens)
                next_token = logits[:, -1].argmax(dim=-1, keepdim=True)
                tokens = torch.cat([tokens, next_token], dim=1)
                
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                elapsed = time.perf_counter() - start
                token_times.append(elapsed)
            
            first_token_times.append(token_times[0])
            all_times.append(token_times)
        
        all_times = np.array(all_times)
        
        result = {
            "prompt_len": prompt_len,
            "gen_len": gen_len,
            "avg_first_token_ms": np.mean(first_token_times) * 1000,
            "avg_per_token_ms": np.mean(all_times) * 1000,
            "median_per_token_ms": np.median(all_times) * 1000,
            "p95_per_token_ms": np.percentile(all_times, 95) * 1000,
            "p99_per_token_ms": np.percentile(all_times, 99) * 1000,
            "total_gen_time_sec": np.mean(all_times.sum(axis=1)),
            "gen_tokens_per_sec": gen_len / np.mean(all_times.sum(axis=1)),
            "status": "success",
        }
        
        logger.info(
            f"  Generation: {result['gen_tokens_per_sec']:.1f} tok/s | "
            f"First token: {result['avg_first_token_ms']:.1f}ms | "
            f"Per token: {result['avg_per_token_ms']:.1f}ms"
        )
    
    except torch.cuda.OutOfMemoryError:
        logger.warning("  Generation benchmark: OOM!")
        result = {"status": "oom"}
        torch.cuda.empty_cache()
    
    return result


def benchmark_model(
    arch: str,
    model: nn.Module,
    vocab_size: int,
    output_dir: str,
    seq_lengths: List[int] = [128, 256, 512, 1024, 2048, 4096],
):
    """Run all benchmarks for a single model."""
    device = get_device()
    model = model.to(device)
    model.eval()
    
    logger.info(f"\n{'='*60}")
    logger.info(f"BENCHMARKING: {arch.upper()}")
    logger.info(f"{'='*60}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Forward pass throughput
    logger.info("\n[1/3] Forward pass throughput...")
    throughput_results = benchmark_throughput(
        model, vocab_size, device,
        seq_lengths=seq_lengths,
        batch_size=8,
        n_warmup=3,
        n_measure=10,
    )
    
    # 2. Single-sample throughput (no batching)
    logger.info("\n[2/3] Single-sample throughput...")
    single_throughput = benchmark_throughput(
        model, vocab_size, device,
        seq_lengths=seq_lengths,
        batch_size=1,
        n_warmup=3,
        n_measure=10,
    )
    
    # 3. Generation latency
    logger.info("\n[3/3] Generation latency...")
    gen_latency = benchmark_generation_latency(
        model, vocab_size, device,
        prompt_len=32,
        gen_len=128,
        batch_size=1,
        n_runs=5,
    )
    
    benchmark_results = {
        "architecture": arch,
        "total_params": model.count_params(),
        "batch_throughput": throughput_results,
        "single_throughput": single_throughput,
        "generation_latency": gen_latency,
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
    }
    
    with open(os.path.join(output_dir, "benchmark.json"), "w") as f:
        json.dump(benchmark_results, f, indent=2)
    
    logger.info(f"Benchmark results saved to {output_dir}/benchmark.json")
    
    return benchmark_results


def main():
    parser = argparse.ArgumentParser(description="Benchmark a language model")
    parser.add_argument("--arch", type=str, required=True,
                       choices=["transformer", "hrm", "mamba", "hybrid"])
    parser.add_argument("--vocab-size", type=int, default=3919)
    parser.add_argument("--checkpoint", type=str, default=None,
                       help="Path to model checkpoint (optional)")
    parser.add_argument("--output-dir", type=str, default="results")
    parser.add_argument("--seq-lengths", type=int, nargs="+",
                       default=[128, 256, 512, 1024, 2048, 4096])
    
    args = parser.parse_args()
    
    from models import create_model
    
    model = create_model(args.arch, args.vocab_size)
    
    if args.checkpoint and os.path.exists(args.checkpoint):
        state_dict = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        for key in ("model_state", "model", "model_state_dict"):
            if isinstance(state_dict, dict) and key in state_dict:
                state_dict = state_dict[key]
                break
        model.load_state_dict(state_dict, strict=False)
        logger.info(f"Loaded checkpoint from {args.checkpoint}")
    elif args.checkpoint:
        logger.warning(f"Checkpoint {args.checkpoint} not found; running benchmark on initialized architecture.")
    
    out_dir = os.path.join(args.output_dir, args.arch)
    benchmark_model(args.arch, model, args.vocab_size, out_dir, args.seq_lengths)


if __name__ == "__main__":
    main()

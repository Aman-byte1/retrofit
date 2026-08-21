"""
Generate Amharic text samples from trained model checkpoints.

Loads:
  • RL-trained Amharic vocabulary (tokenizer/vocab.txt)
  • Model checkpoint (results/<arch>/best_model.pt or results/<arch>_*/best_model.pt)
  • Prompts with diverse Amharic prefixes
  • Evaluates autoregressive generation quality, latency, and tokens/sec
"""

import argparse
import json
import os
import time
from pathlib import Path
from typing import List, Dict, Any, Optional

import torch
import torch.nn.functional as F

from models import create_model


AMHARIC_PROMPTS = [
    "ኢትዮጵያ በምሥራቅ አፍሪካ የምትገኝ",
    "አዲስ አበባ የኢትዮጵያ ዋና ከተማ",
    "የአማርኛ ቋንቋ በሴማዊ የቋንቋዎች ቤተሰብ",
    "የሳይንስና የቴክኖሎጂ እድገት ለሀገር",
    "ታሪክ እንደሚያስረዳው የጥንት",
]


class AmharicTokenizer:
    """Fast longest-prefix subword tokenizer for Amharic generation."""

    def __init__(self, vocab_file: str):
        self.token2id = {}
        self.id2token = {}
        with open(vocab_file, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                token = line.strip()
                if token:
                    self.token2id[token] = idx
                    self.id2token[idx] = token

        self.vocab_size = len(self.token2id)
        self.max_token_len = max(len(t) for t in self.token2id.keys()) if self.token2id else 1
        self.tokens_by_len = {}
        for token, tid in self.token2id.items():
            self.tokens_by_len.setdefault(len(token), {})[token] = tid

    def encode(self, text: str) -> List[int]:
        tokens = []
        i = 0
        n = len(text)
        while i < n:
            matched = False
            max_check = min(self.max_token_len, n - i)
            for length in range(max_check, 0, -1):
                if length in self.tokens_by_len:
                    sub = text[i : i + length]
                    if sub in self.tokens_by_len[length]:
                        tokens.append(self.tokens_by_len[length][sub])
                        i += length
                        matched = True
                        break
            if not matched:
                i += 1
        return tokens

    def decode(self, token_ids: List[int]) -> str:
        return "".join([self.id2token.get(t, "") for t in token_ids])


@torch.no_grad()
def generate_text(
    model: torch.nn.Module,
    tokenizer: AmharicTokenizer,
    prompt: str,
    max_new_tokens: int = 64,
    temperature: float = 0.8,
    top_k: int = 40,
    top_p: float = 0.9,
    device: torch.device = torch.device("cpu"),
) -> Dict[str, Any]:
    """Generate Amharic text from prompt using top-k/top-p sampling."""
    model.eval()
    prompt_ids = tokenizer.encode(prompt)
    if not prompt_ids:
        prompt_ids = [0]

    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    generated_ids = list(prompt_ids)

    start_time = time.perf_counter()

    for _ in range(max_new_tokens):
        # Forward pass
        if input_ids.shape[1] > 512:
            input_context = input_ids[:, -512:]
        else:
            input_context = input_ids

        logits, _ = model(input_context)
        next_token_logits = logits[:, -1, :] / max(1e-5, temperature)

        # Top-k filtering
        if top_k > 0:
            indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
            next_token_logits[indices_to_remove] = -float("Inf")

        # Top-p (nucleus) filtering
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            next_token_logits[0, indices_to_remove] = -float("Inf")

        probs = F.softmax(next_token_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)

        token_id = next_token.item()
        generated_ids.append(token_id)
        input_ids = torch.cat([input_ids, next_token], dim=1)

    elapsed = time.perf_counter() - start_time
    decoded_text = tokenizer.decode(generated_ids)
    continuation = tokenizer.decode(generated_ids[len(prompt_ids):])

    return {
        "prompt": prompt,
        "prompt_tokens": len(prompt_ids),
        "generated_tokens": max_new_tokens,
        "full_text": decoded_text,
        "continuation": continuation,
        "time_seconds": round(elapsed, 4),
        "tokens_per_sec": round(max_new_tokens / max(1e-5, elapsed), 1),
    }


def find_checkpoint(results_dir: str, arch: str) -> Optional[str]:
    """Find checkpoint file for given architecture."""
    candidates = [
        os.path.join(results_dir, arch, "best_model.pt"),
        os.path.join(results_dir, arch, "checkpoint_epoch_1.pt"),
    ]
    res_path = Path(results_dir)
    if res_path.exists():
        for d in sorted(res_path.glob(f"{arch}_*")):
            candidates.extend([
                str(d / "best_model.pt"),
                str(d / "checkpoint_epoch_1.pt"),
            ])
    for c in candidates:
        if os.path.isfile(c):
            return c
    return None


def run_sample_generation(
    results_dir: str = "results",
    tokenizer_dir: str = "tokenizer",
    output_dir: str = "results/analysis",
    max_tokens: int = 50,
):
    """Run generation across all available trained models and export samples."""
    vocab_path = os.path.join(tokenizer_dir, "vocab.txt")
    if not os.path.exists(vocab_path):
        raise FileNotFoundError(f"Vocabulary not found at {vocab_path}")

    tokenizer = AmharicTokenizer(vocab_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Generating Amharic samples on {device} (vocab size: {tokenizer.vocab_size:,})...")

    os.makedirs(output_dir, exist_ok=True)
    all_samples = {}

    for arch in ["transformer", "hrm", "mamba", "hybrid"]:
        ckpt_path = find_checkpoint(results_dir, arch)
        if not ckpt_path:
            print(f"  • {arch.upper()}: No checkpoint found, skipping.")
            continue

        print(f"\n============================================================")
        print(f"  Generating samples for: {arch.upper()} (from {ckpt_path})")
        print(f"============================================================")

        model = create_model(arch, vocab_size=tokenizer.vocab_size).to(device)
        try:
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            state_dict = ckpt.get("model_state", ckpt.get("model", ckpt.get("model_state_dict", ckpt)))
            model.load_state_dict(state_dict, strict=False)
            print(f"  ✓ Loaded weights successfully.")
        except Exception as e:
            print(f"  ⚠️ Warning loading weights: {e}")

        model.eval()
        arch_samples = []

        for p_idx, prompt in enumerate(AMHARIC_PROMPTS):
            sample = generate_text(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                max_new_tokens=max_tokens,
                temperature=0.8,
                top_k=40,
                top_p=0.9,
                device=device,
            )
            arch_samples.append(sample)
            print(f"\n[Prompt {p_idx + 1}]: {prompt}")
            print(f"👉 Continuation: {sample['continuation']}")
            print(f"⚡ Generation speed: {sample['tokens_per_sec']} tokens/sec")

        all_samples[arch] = {
            "arch": arch,
            "checkpoint": ckpt_path,
            "samples": arch_samples,
        }

    # Save to JSON and Markdown
    json_path = os.path.join(output_dir, "generated_samples.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_samples, f, indent=2, ensure_ascii=False)

    md_path = os.path.join(output_dir, "generated_samples.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Qualitative Amharic Generation Samples\n\n")
        f.write("Generated text completions across 50M-parameter models trained on Amharic Wikipedia:\n\n")
        for arch, data in all_samples.items():
            f.write(f"## Architecture: {arch.upper()}\n\n")
            for idx, s in enumerate(data["samples"]):
                f.write(f"### Prompt {idx + 1}: *{s['prompt']}*\n")
                f.write(f"> **Generated Completion:** {s['continuation']}\n\n")
                f.write(f"- **Full Text:** {s['full_text']}\n")
                f.write(f"- **Tokens Generated:** {s['generated_tokens']} tokens in {s['time_seconds']}s ({s['tokens_per_sec']} tok/s)\n\n")
            f.write("---\n\n")

    print(f"\n✓ Generated samples saved to:")
    print(f"  • {json_path}")
    print(f"  • {md_path}")
    return all_samples


def main():
    parser = argparse.ArgumentParser(description="Generate Amharic text from trained checkpoints")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--tokenizer-dir", default="tokenizer")
    parser.add_argument("--output-dir", default="results/analysis")
    parser.add_argument("--max-tokens", type=int, default=50)
    args = parser.parse_args()

    run_sample_generation(
        results_dir=args.results_dir,
        tokenizer_dir=args.tokenizer_dir,
        output_dir=args.output_dir,
        max_tokens=args.max_tokens,
    )


if __name__ == "__main__":
    main()

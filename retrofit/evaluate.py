"""
Evaluation pipeline for voice cloning quality.

Evaluates synthesized speech against reference audio using:
- CER (Character Error Rate) via Whisper
- Speaker Similarity via ECAPA-TDNN
- Combined Score (IWSLT 2026 metric)

Usage:
    # Evaluate a trained model
    python -m retrofit.evaluate --method uniform_lora --adapter-path experiments/uniform_lora_r8/adapter_best.pt

    # Evaluate zero-shot baseline
    python -m retrofit.evaluate --method zero_shot

    # Evaluate specific language
    python -m retrofit.evaluate --method uniform_lora --language fr
"""

import argparse
import logging
import sys
import yaml
import torch
import json
import numpy as np
from pathlib import Path
from typing import Dict, Optional
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("retrofit.evaluate")


def run_evaluation(
    model=None,
    config: Optional[Dict] = None,
    output_dir: Optional[Path] = None,
    method_name: str = "unknown",
    languages: Optional[list] = None,
    max_samples: Optional[int] = None,
    adapter_path: Optional[str] = None,
) -> Dict:
    """
    Run the full evaluation pipeline.
    
    Args:
        model: RetrofitModel instance (if None, creates one from config)
        config: Configuration dict
        output_dir: Where to save results
        method_name: Name of the adaptation method being evaluated
        languages: Languages to evaluate (None = all from config)
        max_samples: Max samples per language
        adapter_path: Path to adapter weights to load
        
    Returns:
        Dict of results per language
    """
    from .metrics import SpeakerSimilarityScorer, CERScorer, MetricsAggregator
    from .data import IWSLTEvalDataset
    
    if config is None:
        config = {}
    
    if output_dir is None:
        output_dir = Path(config.get("experiment", {}).get("output_dir", "./experiments"))
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize model if not provided
    if model is None:
        from .model import RetrofitModel
        model_config = {
            **config.get("model", {}),
            "lora": config.get("lora", {}),
            "adaptation_method": method_name,
        }
        model = RetrofitModel(model_config)
        
        if adapter_path:
            model.load_adapters(adapter_path)
    
    model.eval_mode()
    
    # Initialize scorers
    speaker_cfg = config.get("speaker_encoder", {})
    asr_cfg = config.get("asr", {})
    
    speaker_scorer = SpeakerSimilarityScorer(
        model_name=speaker_cfg.get("model", "speechbrain/spkrec-ecapa-voxceleb"),
        device=config.get("model", {}).get("device", "cuda"),
    )
    
    cer_scorer = CERScorer(
        model_size=asr_cfg.get("model", "large-v3"),
        device=config.get("model", {}).get("device", "cuda"),
        compute_type=asr_cfg.get("compute_type", "float16"),
    )
    
    # Determine languages
    if languages is None:
        languages = config.get("data", {}).get("eval_languages", ["fr", "ar", "zh"])
    
    # Language code mapping for Whisper
    lang_map = {"fr": "fr", "ar": "ar", "zh": "zh"}
    
    # Run evaluation
    aggregator = MetricsAggregator()
    all_results = {}
    
    for lang in languages:
        logger.info(f"\n{'='*40}")
        logger.info(f"Evaluating language: {lang}")
        logger.info(f"{'='*40}")
        
        # Load evaluation data
        eval_dataset = IWSLTEvalDataset(
            language=lang,
            target_sr=config.get("audio", {}).get("sample_rate", 24000),
            max_samples=max_samples,
        )
        
        lang_cer_scores = []
        lang_sim_scores = []
        lang_combined_scores = []
        
        for idx in tqdm(range(len(eval_dataset)), desc=f"Eval {lang}"):
            sample = eval_dataset[idx]
            
            try:
                # Synthesize speech using the model
                gen_audio, gen_sr = model.synthesize(
                    text=sample["text"],
                    ref_audio=sample["ref_audio"],
                    ref_text="",  # We don't have ref_text in this dataset
                    ref_sr=sample["ref_sr"],
                )
                
                # Compute CER
                cer = cer_scorer.compute_cer(
                    audio=gen_audio,
                    reference_text=sample["text"],
                    sr=gen_sr,
                    language=lang_map.get(lang, lang),
                )
                
                # Compute speaker similarity
                speaker_sim = speaker_scorer.compute_similarity(
                    audio_ref=sample["ref_audio"],
                    audio_gen=gen_audio,
                    sr=16000,  # Speaker encoder expects 16kHz
                )
                
                # Record
                aggregator.add(
                    sample_id=sample["id"],
                    language=lang,
                    cer=cer,
                    speaker_similarity=speaker_sim,
                    method=method_name,
                )
                
                lang_cer_scores.append(cer)
                lang_sim_scores.append(speaker_sim)
                lang_combined_scores.append(0.5 * (1 - cer) + 0.5 * speaker_sim)
                
            except Exception as e:
                logger.warning(f"Failed on sample {sample['id']}: {e}")
                continue
        
        # Language summary
        if lang_cer_scores:
            avg_cer = np.mean(lang_cer_scores)
            avg_sim = np.mean(lang_sim_scores)
            avg_combined = np.mean(lang_combined_scores)
            
            all_results[lang] = {
                "count": len(lang_cer_scores),
                "avg_cer": float(avg_cer),
                "avg_speaker_sim": float(avg_sim),
                "avg_combined": float(avg_combined),
                "std_cer": float(np.std(lang_cer_scores)),
                "std_speaker_sim": float(np.std(lang_sim_scores)),
            }
            
            logger.info(f"  {lang}: CER={avg_cer:.3f} ± {np.std(lang_cer_scores):.3f} | "
                       f"SpeakerSim={avg_sim:.3f} ± {np.std(lang_sim_scores):.3f} | "
                       f"Combined={avg_combined:.3f}")
    
    # Save results
    results_path = output_dir / f"eval_results_{method_name}.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"Results saved to {results_path}")
    
    # Save detailed per-sample results
    csv_path = output_dir / f"eval_detailed_{method_name}.csv"
    aggregator.save(str(csv_path))
    
    return all_results


def main():
    parser = argparse.ArgumentParser(description="Retrofit: Voice Cloning Evaluation")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--method", type=str, default="zero_shot",
                       choices=["zero_shot", "uniform_lora", "targeted_lora", "full_finetune"])
    parser.add_argument("--adapter-path", type=str, default=None,
                       help="Path to adapter weights")
    parser.add_argument("--language", type=str, nargs="+", default=None,
                       help="Languages to evaluate")
    parser.add_argument("--max-samples", type=int, default=None,
                       help="Max samples per language")
    parser.add_argument("--output-dir", type=str, default=None)
    
    args = parser.parse_args()
    
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    
    output_dir = Path(args.output_dir or config["experiment"]["output_dir"]) / args.method
    
    results = run_evaluation(
        config=config,
        output_dir=output_dir,
        method_name=args.method,
        languages=args.language,
        max_samples=args.max_samples,
        adapter_path=args.adapter_path,
    )
    
    # Print final summary
    print("\n" + "=" * 60)
    print(f"EVALUATION SUMMARY — Method: {args.method}")
    print("=" * 60)
    
    for lang, stats in results.items():
        print(f"  {lang.upper()}: CER={stats['avg_cer']:.3f} | "
              f"SpkSim={stats['avg_speaker_sim']:.3f} | "
              f"Combined={stats['avg_combined']:.3f} | "
              f"N={stats['count']}")
    
    # Overall
    if results:
        overall_cer = np.mean([s["avg_cer"] for s in results.values()])
        overall_sim = np.mean([s["avg_speaker_sim"] for s in results.values()])
        overall_comb = np.mean([s["avg_combined"] for s in results.values()])
        print(f"\n  OVERALL: CER={overall_cer:.3f} | SpkSim={overall_sim:.3f} | Combined={overall_comb:.3f}")


if __name__ == "__main__":
    main()

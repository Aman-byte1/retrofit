"""
Evaluation for the Retrofit Voice Cloner.

Generates speech using the retrofitted model, then measures:
- CER (via Whisper): Is the speech intelligible?
- Speaker Similarity (via ECAPA-TDNN): Does it sound like the reference speaker?
- Combined Score (IWSLT metric): Overall quality

Usage:
    # Evaluate with adapter
    python -m retrofit.evaluate --language fr --adapter-path experiments/retrofit_film_fr/adapter_best.pt

    # Evaluate without adapter (baseline — just the TTS model, no cloning)
    python -m retrofit.evaluate --language fr --no-adapter

    # Evaluate all languages
    python -m retrofit.evaluate --language fr ar zh --adapter-path experiments/retrofit_film_fr/adapter_best.pt
"""

import argparse
import logging
import sys
import json
import yaml
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Optional, List
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
    method_name: str = "retrofit",
    languages: Optional[List[str]] = None,
    max_samples: Optional[int] = None,
    adapter_path: Optional[str] = None,
) -> Dict:
    """
    Run the full evaluation pipeline.
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
        from .model import RetrofitVoiceCloner
        model_config = {
            **config.get("model", {}),
            "adapter": config.get("adapter", {}),
            "speaker_encoder": config.get("speaker_encoder", {}),
        }
        model = RetrofitVoiceCloner(model_config)
        
        if adapter_path:
            model.load_adapter(adapter_path)
    
    model.eval_mode()
    
    # Initialize scorers
    speaker_scorer = SpeakerSimilarityScorer(
        model_name=config.get("speaker_encoder", {}).get(
            "model", "speechbrain/spkrec-ecapa-voxceleb"
        ),
        device=config.get("model", {}).get("device", "cuda"),
    )
    
    asr_cfg = config.get("asr", {})
    cer_scorer = CERScorer(
        model_size=asr_cfg.get("model", "large-v3"),
        device=config.get("model", {}).get("device", "cuda"),
        compute_type=asr_cfg.get("compute_type", "float16"),
    )
    
    # Languages to evaluate
    if languages is None:
        languages = config.get("data", {}).get("eval_languages", ["fr", "ar", "zh"])
    
    aggregator = MetricsAggregator()
    all_results = {}
    
    for lang in languages:
        logger.info(f"\n{'='*40}")
        logger.info(f"Evaluating: {lang.upper()}")
        logger.info(f"{'='*40}")
        
        # Switch TTS language
        try:
            model.switch_language(lang)
        except Exception as e:
            logger.warning(f"Could not switch to {lang}: {e}. Skipping.")
            continue
        
        # Load eval data
        eval_dataset = IWSLTEvalDataset(
            language=lang,
            target_sr=config.get("audio", {}).get("sample_rate", 24000),
            max_samples=max_samples,
        )
        
        cer_scores, sim_scores, combined_scores = [], [], []
        
        for idx in tqdm(range(len(eval_dataset)), desc=f"Eval {lang}"):
            sample = eval_dataset[idx]
            
            try:
                # Generate speech with voice cloning
                gen_audio, gen_sr = model.synthesize(
                    text=sample["text"],
                    ref_audio=sample["ref_audio"],
                    ref_sr=sample["ref_sr"],
                )
                
                # Compute CER
                cer = cer_scorer.compute_cer(
                    audio=gen_audio,
                    reference_text=sample["text"],
                    sr=gen_sr,
                    language=lang,
                )
                
                # Compute speaker similarity
                sim = speaker_scorer.compute_similarity(
                    audio_ref=sample["ref_audio"],
                    audio_gen=gen_audio,
                    sr=16000,
                )
                
                combined = 0.5 * (1 - min(cer, 1.0)) + 0.5 * sim
                
                aggregator.add(
                    sample_id=sample["id"],
                    language=lang,
                    cer=cer,
                    speaker_similarity=sim,
                    method=method_name,
                )
                
                cer_scores.append(cer)
                sim_scores.append(sim)
                combined_scores.append(combined)
                
            except Exception as e:
                logger.warning(f"Failed on {sample['id']}: {e}")
                continue
        
        if cer_scores:
            result = {
                "count": len(cer_scores),
                "avg_cer": float(np.mean(cer_scores)),
                "avg_speaker_sim": float(np.mean(sim_scores)),
                "avg_combined": float(np.mean(combined_scores)),
                "std_cer": float(np.std(cer_scores)),
                "std_speaker_sim": float(np.std(sim_scores)),
            }
            all_results[lang] = result
            
            logger.info(
                f"  {lang}: CER={result['avg_cer']:.3f} | "
                f"SpkSim={result['avg_speaker_sim']:.3f} | "
                f"Combined={result['avg_combined']:.3f} | N={result['count']}"
            )
    
    # Save results
    with open(output_dir / f"eval_results_{method_name}.json", "w") as f:
        json.dump(all_results, f, indent=2)
    
    aggregator.save(str(output_dir / f"eval_detailed_{method_name}.csv"))
    
    return all_results


def main():
    parser = argparse.ArgumentParser(description="Retrofit: Evaluate Voice Cloning")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--language", nargs="+", default=["fr"])
    parser.add_argument("--adapter-path", default=None)
    parser.add_argument("--no-adapter", action="store_true",
                       help="Run without adapter (baseline: TTS only, no cloning)")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
    
    args = parser.parse_args()
    
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    method = "baseline_no_cloning" if args.no_adapter else "retrofit"
    output_dir = Path(args.output_dir or config["experiment"]["output_dir"]) / method
    
    results = run_evaluation(
        config=config,
        output_dir=output_dir,
        method_name=method,
        languages=args.language,
        max_samples=args.max_samples,
        adapter_path=None if args.no_adapter else args.adapter_path,
    )
    
    # Final summary
    print("\n" + "=" * 60)
    print(f"RESULTS — {method}")
    print("=" * 60)
    for lang, stats in results.items():
        print(f"  {lang.upper()}: CER={stats['avg_cer']:.3f} | "
              f"SpkSim={stats['avg_speaker_sim']:.3f} | "
              f"Combined={stats['avg_combined']:.3f}")
    
    if results:
        overall = {k: np.mean([s[k] for s in results.values()])
                  for k in ["avg_cer", "avg_speaker_sim", "avg_combined"]}
        print(f"\n  OVERALL: CER={overall['avg_cer']:.3f} | "
              f"SpkSim={overall['avg_speaker_sim']:.3f} | "
              f"Combined={overall['avg_combined']:.3f}")


if __name__ == "__main__":
    main()

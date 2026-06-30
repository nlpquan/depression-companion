"""Ablation study script for comparing fusion methods."""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from loguru import logger
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from depression_companion.models.cross_modal_attention import (
    CrossModalFusion,
    SimpleConcatFusion,
)
from depression_companion.models.multi_task_model import MultiTaskDepressionModel


def run_ablation(config: dict[str, Any]) -> dict[str, dict[str, float]]:
    """Run ablation study comparing fusion methods.
    
    Args:
        config: Experiment configuration.
        
    Returns:
        Dictionary mapping variant name to metrics.
    """
    logger.info("=" * 60)
    logger.info("Running Ablation Study")
    logger.info("=" * 60)
    
    results = {}
    
    # Generate synthetic data for demonstration
    np.random.seed(config["experiment"]["seed"])
    torch.manual_seed(config["experiment"]["seed"])
    
    n_samples = 200
    audio_dim = config["model"]["audio_dim"]
    text_dim = config["model"]["text_dim"]
    
    # Synthetic features
    audio_features = torch.randn(n_samples, audio_dim)
    text_features = torch.randn(n_samples, text_dim)
    
    # Synthetic labels (with some correlation to features)
    # Make audio slightly more predictive for depression
    audio_signal = audio_features[:, :10].mean(dim=1)
    text_signal = text_features[:, :10].mean(dim=1)
    combined_signal = 0.6 * audio_signal + 0.4 * text_signal
    labels = (torch.sigmoid(combined_signal) > 0.5).long()
    
    for variant in config["ablation"]["variants"]:
        name = variant["name"]
        logger.info(f"\n--- Variant: {name} ---")
        logger.info(f"Description: {variant['description']}")
        
        use_audio = variant["use_audio"]
        use_text = variant["use_text"]
        fusion_method = variant["fusion"]
        
        # Create model based on variant
        if fusion_method == "cross_modal":
            model = CrossModalFusion(
                audio_dim=audio_dim,
                text_dim=text_dim,
                hidden_dim=config["model"]["hidden_dim"],
                num_heads=config["model"]["num_heads"],
                num_fusion_layers=config["model"]["num_fusion_layers"],
                dropout=config["model"]["dropout"],
            )
        elif fusion_method == "concat":
            model = SimpleConcatFusion(audio_dim, text_dim)
        else:
            model = None
        
        # Compute features for this variant
        variant_audio = audio_features if use_audio else torch.randn_like(audio_features) * 0.01
        variant_text = text_features if use_text else torch.randn_like(text_features) * 0.01
        
        if model is not None:
            with torch.no_grad():
                output = model(variant_audio, variant_text)
                fused_features = output["fused_features"]
        elif use_audio:
            fused_features = variant_audio
        elif use_text:
            fused_features = variant_text
        else:
            continue
        
        # Simple classifier on fused features
        classifier = torch.nn.Linear(fused_features.shape[-1], 2)
        with torch.no_grad():
            logits = classifier(fused_features)
            preds = logits.argmax(dim=-1)
        
        # Compute metrics
        metrics = {
            "accuracy": accuracy_score(labels.numpy(), preds.numpy()),
            "f1": f1_score(labels.numpy(), preds.numpy(), average="macro"),
        }
        
        try:
            probs = torch.softmax(logits, dim=-1)[:, 1]
            metrics["auc"] = roc_auc_score(labels.numpy(), probs.numpy())
        except ValueError:
            metrics["auc"] = 0.5
        
        results[name] = metrics
        
        logger.info(f"Results: {metrics}")
    
    # Print comparison table
    logger.info("\n" + "=" * 60)
    logger.info("Ablation Results Summary")
    logger.info("=" * 60)
    logger.info(f"{'Variant':<25} {'Accuracy':<10} {'F1':<10} {'AUC':<10}")
    logger.info("-" * 55)
    
    for name, metrics in results.items():
        logger.info(
            f"{name:<25} {metrics['accuracy']:.4f}     "
            f"{metrics['f1']:.4f}     {metrics['auc']:.4f}"
        )
    
    # Save results
    output_path = Path("results") / config["experiment"]["name"]
    output_path.mkdir(parents=True, exist_ok=True)
    
    with open(output_path / "ablation_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"\nResults saved to {output_path / 'ablation_results.json'}")
    
    return results


def main() -> None:
    """Run ablation study."""
    parser = argparse.ArgumentParser(description="Run ablation study")
    parser.add_argument(
        "--config",
        type=str,
        default="config/experiments/cross_modal.yaml",
        help="Path to experiment config",
    )
    args = parser.parse_args()
    
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    
    run_ablation(config)
    
    logger.info("Ablation study complete!")


if __name__ == "__main__":
    main()
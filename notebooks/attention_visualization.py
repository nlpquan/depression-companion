"""
Attention Weight Visualization
==============================
Visualize cross-modal attention weights between audio and text features.
Run as: python notebooks/attention_visualization.py
Or copy cells into Jupyter notebook.
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch

# Add src to path
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from depression_companion.models.cross_modal_attention import CrossModalFusion

# Set style
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 8)
plt.rcParams["font.size"] = 12


def plot_attention_heatmap(
    attention_weights: np.ndarray,
    title: str = "Cross-Modal Attention",
    x_label: str = "Audio Features",
    y_label: str = "Text Features",
    save_path: str = "attention_heatmap.png",
) -> None:
    """Plot attention heatmap.
    
    Args:
        attention_weights: Attention matrix (text_len, audio_len).
        title: Plot title.
        x_label: X-axis label.
        y_label: Y-axis label.
        save_path: Path to save the figure.
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
    sns.heatmap(
        attention_weights,
        cmap="YlOrRd",
        annot=False,
        cbar_kws={"label": "Attention Weight"},
        ax=ax,
    )
    
    ax.set_title(title, fontsize=16, fontweight="bold")
    ax.set_xlabel(x_label, fontsize=14)
    ax.set_ylabel(y_label, fontsize=14)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved: {save_path}")


def plot_feature_importance(
    importance_dict: dict[str, float],
    title: str = "Feature Importance from Attention",
    top_k: int = 20,
    save_path: str = "feature_importance.png",
) -> None:
    """Plot feature importance bar chart.
    
    Args:
        importance_dict: Dictionary of feature_name -> importance_score.
        title: Plot title.
        top_k: Number of top features to show.
        save_path: Path to save the figure.
    """
    # Sort and take top k
    sorted_features = sorted(
        importance_dict.items(),
        key=lambda x: x[1],
        reverse=True,
    )[:top_k]
    
    names, scores = zip(*sorted_features)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    colors = plt.cm.Reds([0.3 + 0.7 * (s / max(scores)) for s in scores])
    bars = ax.barh(range(len(names)), scores, color=colors)
    
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.set_xlabel("Importance Score", fontsize=12)
    ax.set_title(title, fontsize=16, fontweight="bold")
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved: {save_path}")


def plot_ablation_comparison(
    results: dict[str, dict[str, float]],
    metric: str = "auc",
    save_path: str = "ablation_comparison.png",
) -> None:
    """Plot ablation study comparison.
    
    Args:
        results: Dictionary of variant_name -> metrics_dict.
        metric: Metric to compare.
        save_path: Path to save the figure.
    """
    names = list(results.keys())
    values = [results[n][metric] for n in names]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = ["#3498db", "#3498db", "#e74c3c", "#2ecc71"]
    bars = ax.bar(names, values, color=colors[:len(names)])
    
    # Add value labels on bars
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{val:.4f}",
            ha="center",
            fontweight="bold",
            fontsize=12,
        )
    
    ax.set_ylabel(metric.upper(), fontsize=14)
    ax.set_title(f"Ablation Study: {metric.upper()} by Fusion Method", fontsize=16, fontweight="bold")
    ax.set_ylim(0, max(values) * 1.2)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved: {save_path}")


def main() -> None:
    """Run all visualizations."""
    print("=" * 60)
    print("Cross-Modal Attention Visualization")
    print("=" * 60)
    
    # Create model
    audio_dim = 768
    text_dim = 768
    hidden_dim = 256
    
    model = CrossModalFusion(
        audio_dim=audio_dim,
        text_dim=text_dim,
        hidden_dim=hidden_dim,
        num_heads=8,
    )
    model.eval()
    
    # Generate sample data
    # Simulate: 10 text features attending to 15 audio features
    audio_features = torch.randn(1, 15, audio_dim)
    text_features = torch.randn(1, 10, text_dim)
    
    with torch.no_grad():
        output = model(audio_features, text_features)
    
    # 1. Plot Text → Audio attention
    print("\n1. Text-to-Audio Attention")
    text_to_audio_attn = output["text_to_audio_attention"]
    if text_to_audio_attn is not None and text_to_audio_attn.numel() > 1:
        plot_attention_heatmap(
            text_to_audio_attn.squeeze(0).numpy(),
            title="Text Features Attending to Audio Features",
            x_label="Audio Features (Time/Mel Bands)",
            y_label="Text Features (Tokens)",
            save_path="results/text_to_audio_attention.png",
        )
    
    # 2. Plot Audio → Text attention
    print("\n2. Audio-to-Text Attention")
    audio_to_text_attn = output["audio_to_text_attention"]
    if audio_to_text_attn is not None and audio_to_text_attn.numel() > 1:
        plot_attention_heatmap(
            audio_to_text_attn.squeeze(0).numpy(),
            title="Audio Features Attending to Text Features",
            x_label="Text Features (Tokens)",
            y_label="Audio Features (Time/Mel Bands)",
            save_path="results/audio_to_text_attention.png",
        )
    
    # 3. Feature importance from attention
    print("\n3. Feature Importance")
    if audio_to_text_attn is not None and audio_to_text_attn.numel() > 1:
        # Simulated feature names
        audio_feature_names = [
            "MFCC_1", "MFCC_2", "MFCC_3", "Pitch", "Energy",
            "Spectral_Centroid", "Zero_Crossing", "Harmonics",
            "Jitter", "Shimmer", "Loudness", "Speaking_Rate",
            "Pause_Duration", "Voice_Breaks", "F0_Variability",
        ]
        
        # Average attention weights over text queries
        avg_attention = audio_to_text_attn.mean(dim=1).squeeze(0)  # (audio_len,)
        
        if len(avg_attention) <= len(audio_feature_names):
            importance = {
                audio_feature_names[i]: avg_attention[i].item()
                for i in range(len(avg_attention))
            }
            
            plot_feature_importance(
                importance,
                title="Audio Feature Importance for Depression Detection",
                top_k=10,
                save_path="results/audio_feature_importance.png",
            )
    
    # 4. Ablation comparison (using sample results)
    print("\n4. Ablation Study Results")
    sample_ablation_results = {
        "Audio Only": {"accuracy": 0.68, "f1": 0.64, "auc": 0.72},
        "Text Only": {"accuracy": 0.71, "f1": 0.68, "auc": 0.76},
        "Simple Concat": {"accuracy": 0.75, "f1": 0.72, "auc": 0.80},
        "Cross-Attention": {"accuracy": 0.82, "f1": 0.79, "auc": 0.86},
    }
    
    plot_ablation_comparison(
        sample_ablation_results,
        metric="auc",
        save_path="results/ablation_comparison.png",
    )
    
    print("\n" + "=" * 60)
    print("All visualizations saved to results/ directory")
    print("=" * 60)


if __name__ == "__main__":
    # Create results directory
    Path("results").mkdir(exist_ok=True)
    main()
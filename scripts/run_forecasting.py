"""Run mood forecasting and early warning system demonstration."""

import sys
from pathlib import Path

import numpy as np
import torch
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from depression_companion.models.forecasting import MoodForecaster
from depression_companion.models.early_warning import (
    EarlyWarningSystem,
    RelapseRiskClassifier,
)


def generate_synthetic_mood_data(
    n_days: int = 60,
    n_features: int = 10,
    seed: int = 42,
) -> np.ndarray:
    """Generate synthetic mood tracking data.
    
    Args:
        n_days: Number of days of data.
        n_features: Number of features per day.
        seed: Random seed.
        
    Returns:
        Array of shape (n_days, n_features).
    """
    np.random.seed(seed)
    
    # Base mood pattern (weekly cycle)
    t = np.arange(n_days)
    base_mood = 0.5 + 0.1 * np.sin(2 * np.pi * t / 7)  # Weekly cycle
    
    # Add trend (slow decline then recovery)
    trend = -0.002 * t + 0.00003 * t**2  # Quadratic trend
    
    # Add noise
    noise = np.random.randn(n_days) * 0.1
    
    mood = base_mood + trend + noise
    mood = np.clip(mood, 0, 1)
    
    # Generate correlated features
    features = np.zeros((n_days, n_features))
    features[:, 0] = mood  # Primary mood feature
    
    # Other features correlated with mood
    for i in range(1, n_features):
        correlation = np.random.uniform(0.3, 0.9)
        features[:, i] = correlation * mood + (1 - correlation) * np.random.randn(n_days) * 0.15
        features[:, i] = np.clip(features[:, i], 0, 1)
    
    return features.astype(np.float32)


def run_forecasting_demo() -> None:
    """Demonstrate forecasting and early warning system."""
    logger.info("=" * 60)
    logger.info("Mood Forecasting & Early Warning Demo")
    logger.info("=" * 60)
    
    # Generate synthetic data
    n_days = 60
    n_features = 10
    data = generate_synthetic_mood_data(n_days, n_features)
    
    logger.info(f"Generated {n_days} days of data with {n_features} features")
    
    # --- Forecasting ---
    logger.info("\n--- Mood Forecasting ---")
    
    model = MoodForecaster(
        input_dim=n_features,
        hidden_dim=64,
        forecast_horizons=[1, 3, 7],
    )
    model.eval()
    
    # Use last 14 days to forecast
    history_length = 14
    historical = torch.from_numpy(data[-history_length:]).unsqueeze(0)  # (1, 14, 10)
    
    with torch.no_grad():
        trajectory = model.predict_trajectory(historical)
    
    logger.info("Mood Forecast (next 7 days):")
    for horizon in [1, 3, 7]:
        pred = trajectory["trajectory"][f"day_{horizon}"].item()
        lower = trajectory["lower_bound"][f"day_{horizon}"].item()
        upper = trajectory["upper_bound"][f"day_{horizon}"].item()
        logger.info(f"  Day {horizon}: {pred:.3f} [{lower:.3f}, {upper:.3f}]")
    
    # --- Early Warning System ---
    logger.info("\n--- Early Warning System ---")
    
    ews = EarlyWarningSystem(baseline_window=14)
    
    # Create feature dictionary from data
    historical_dict = {}
    indicator_names = [
        "sleep_disruption", "social_withdrawal", "speech_energy_drop",
        "activity_reduction", "mood_variability", "negative_sentiment_spike",
    ]
    
    for i, name in enumerate(indicator_names):
        if i < n_features:
            historical_dict[name] = data[:, i]
    
    # Compute baselines
    baselines = ews.compute_baseline(historical_dict)
    
    # Current features (last day)
    current = {name: float(data[-1, i]) for i, name in enumerate(indicator_names) if i < n_features}
    
    # Detect warnings
    warnings = ews.detect_warnings(current, baselines)
    
    logger.info(f"Risk Level: {warnings['risk_level'].upper()}")
    logger.info(f"Risk Score: {warnings['risk_score']:.3f}")
    logger.info(f"Active Warnings: {warnings['num_active_warnings']}")
    
    if warnings["active_warnings"]:
        logger.info("\nActive warning signals:")
        for w in warnings["active_warnings"]:
            logger.info(f"  • {w['description']} (Lead time: {w['lead_time_days']} days)")
    
    # --- Relapse Risk ---
    logger.info("\n--- Relapse Risk Classification ---")
    
    risk_model = RelapseRiskClassifier(input_dim=n_features, hidden_dim=64)
    risk_model.eval()
    
    # Use aggregated features (mean and std over last 7 days)
    recent = torch.from_numpy(data[-7:]).float()
    agg_features = torch.cat([
        recent.mean(dim=0),
        recent.std(dim=0),
    ]).unsqueeze(0)  # (1, 2*n_features)
    
    # Adjust risk model for concatenated features
    risk_model_adapted = RelapseRiskClassifier(
        input_dim=2 * n_features,
        hidden_dim=64,
    )
    risk_model_adapted.eval()
    
    with torch.no_grad():
        risk_output = risk_model_adapted(agg_features)
    
    risk_labels = ["Low", "Medium", "High"]
    predicted_risk = risk_output["predicted_class"].item()
    
    logger.info(f"Relapse Risk: {risk_labels[predicted_risk]}")
    logger.info(f"Risk Score: {risk_output['risk_score'].item():.3f}")
    logger.info("Class Probabilities:")
    for i, label in enumerate(risk_labels):
        logger.info(f"  {label}: {risk_output['probs'][0, i].item():.3f}")
    
    logger.info("\n" + "=" * 60)
    logger.info("Demo complete!")
    logger.info("=" * 60)


if __name__ == "__main__":
    run_forecasting_demo()
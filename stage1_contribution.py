from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from config import ContributionConfig


@dataclass
class ContributionResult:
    perturbation_sets: dict[str, np.ndarray]
    contributions: dict[str, float]
    high_contribution_features: list[str]
    threshold: float


def js_divergence(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), 1e-12, 1.0)
    q = np.clip(np.asarray(q, dtype=float), 1e-12, 1.0)
    p = p / p.sum(axis=-1, keepdims=True)
    q = q / q.sum(axis=-1, keepdims=True)
    midpoint = 0.5 * (p + q)
    return 0.5 * (
        np.sum(p * np.log(p / midpoint), axis=-1)
        + np.sum(q * np.log(q / midpoint), axis=-1)
    )


class FeatureContributionQuantifier:
    def __init__(
        self,
        model,
        train_frame: pd.DataFrame,
        feature_names: list[str],
        model_inputs: Callable[[pd.DataFrame], np.ndarray],
        config: ContributionConfig,
    ):
        self.model = model
        self.train_frame = train_frame.reset_index(drop=True).copy()
        self.feature_names = feature_names
        self.model_inputs = model_inputs
        self.config = config
        self.baseline = self.train_frame[self.feature_names].median(axis=0).to_frame().T
        self.baseline_distribution = self._predict_distribution(self.baseline)[0]

    def _predict_distribution(self, frame: pd.DataFrame) -> np.ndarray:
        output = self.model.predict(
            self.model_inputs(frame),
            batch_size=self.config.inference_batch_size,
            verbose=0,
        )
        if output.ndim == 1 or output.shape[-1] == 1:
            positive = output.reshape(-1, 1)
            return np.concatenate([1.0 - positive, positive], axis=1)
        return output

    def perturbation_values(self, feature: str) -> np.ndarray:
        values = pd.to_numeric(self.train_frame[feature], errors="coerce").dropna().to_numpy(dtype=float)
        if len(values) == 0:
            return np.array([], dtype=float)
        minimum, maximum = float(values.min()), float(values.max())
        difference = maximum - minimum
        unique = np.unique(values)
        diversity = len(unique) / len(values)
        has_decimal = np.any(~np.isclose(unique, np.round(unique)))

        if difference >= 10_000 and diversity >= self.config.diversity_high:
            # Case 1: min + k * diff / S, k = 0, ..., S - 1.
            return minimum + np.arange(len(values), dtype=float) * difference / len(values)
        if difference >= 10_000 and diversity <= self.config.diversity_low:
            # Case 2: retain observed sparse or discrete values.
            return unique
        if difference <= 1 and has_decimal:
            # Case 3: min + k * diff / |unique|, k = 0, ..., |unique| - 1.
            return minimum + np.arange(len(unique), dtype=float) * difference / len(unique)
        return unique

    def quantify(self) -> tuple[dict[str, np.ndarray], dict[str, float]]:
        perturbation_sets: dict[str, np.ndarray] = {}
        contributions: dict[str, float] = {}
        for feature in self.feature_names:
            values = self.perturbation_values(feature)
            perturbation_sets[feature] = values
            if len(values) == 0:
                contributions[feature] = 0.0
                continue
            largest = 0.0
            for start in range(0, len(values), self.config.inference_batch_size):
                batch_values = values[start : start + self.config.inference_batch_size]
                batch = pd.concat([self.baseline] * len(batch_values), ignore_index=True)
                batch[feature] = batch_values
                divergence = js_divergence(
                    self._predict_distribution(batch),
                    np.repeat(self.baseline_distribution[None, :], len(batch), axis=0),
                )
                largest = max(largest, float(np.max(divergence)))
            contributions[feature] = largest
        return perturbation_sets, contributions


def select_high_contribution_features(
    contributions: dict[str, float], config: ContributionConfig
) -> tuple[list[str], float]:
    names = list(contributions)
    scores = np.asarray([contributions[name] for name in names], dtype=float)
    if len(scores) == 0:
        return [], 0.0
    quantile_threshold = float(np.quantile(scores, config.high_contribution_quantile))
    if config.high_contribution_method == "quantile" or len(np.unique(scores)) < 3:
        threshold = quantile_threshold
    elif config.high_contribution_method == "kde":
        from scipy.stats import gaussian_kde

        grid = np.linspace(float(scores.min()), float(scores.max()), 256)
        density = gaussian_kde(scores)(grid)
        peak = int(np.argmax(density))
        right = np.where(np.diff(np.sign(np.diff(density[peak:]))) > 0)[0]
        threshold = float(grid[peak + right[0] + 1]) if len(right) else quantile_threshold
        threshold = max(threshold, quantile_threshold)
    else:
        raise ValueError("high_contribution_method must be 'quantile' or 'kde'")
    selected = [name for name, score in contributions.items() if score >= threshold]
    return selected, threshold


def run_stage1(
    model,
    train_frame: pd.DataFrame,
    feature_names: list[str],
    model_inputs: Callable[[pd.DataFrame], np.ndarray],
    config: ContributionConfig,
) -> ContributionResult:
    quantifier = FeatureContributionQuantifier(model, train_frame, feature_names, model_inputs, config)
    perturbation_sets, contributions = quantifier.quantify()
    high_features, threshold = select_high_contribution_features(contributions, config)
    return ContributionResult(perturbation_sets, contributions, high_features, threshold)

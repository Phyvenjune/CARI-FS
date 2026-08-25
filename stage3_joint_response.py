from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Callable

import numpy as np
import pandas as pd

from config import JointResponseConfig
from stage1_contribution import js_divergence


@dataclass
class JointResponseResult:
    added_features: list[str]
    final_features: list[str]
    pair_responses: dict[tuple[str, str], float]


class JointResponseSupplementer:
    def __init__(
        self,
        model,
        train_frame: pd.DataFrame,
        all_features: list[str],
        model_inputs: Callable[[pd.DataFrame], np.ndarray],
        contributions: dict[str, float],
        perturbation_sets: dict[str, np.ndarray],
        config: JointResponseConfig,
        seed: int,
    ):
        self.model = model
        self.train_frame = train_frame.reset_index(drop=True)
        self.all_features = all_features
        self.model_inputs = model_inputs
        self.contributions = contributions
        self.perturbation_sets = perturbation_sets
        self.config = config
        self.rng = np.random.default_rng(seed)

    def _distribution(self, frame: pd.DataFrame) -> np.ndarray:
        prediction = self.model.predict(
            self.model_inputs(frame),
            batch_size=self.config.inference_batch_size,
            verbose=0,
        )
        if prediction.ndim == 1 or prediction.shape[-1] == 1:
            positive = prediction.reshape(-1, 1)
            return np.concatenate([1.0 - positive, positive], axis=1)
        return prediction

    def _contexts(self) -> pd.DataFrame:
        count = min(self.config.max_contexts, len(self.train_frame))
        if count == 0:
            return self.train_frame.iloc[:0]
        positions = self.rng.choice(len(self.train_frame), size=count, replace=False)
        return self.train_frame.iloc[np.sort(positions)][self.all_features].reset_index(drop=True)

    def _limited_values(self, feature: str) -> np.ndarray:
        values = np.asarray(self.perturbation_sets.get(feature, []), dtype=float)
        if len(values) <= self.config.max_values_per_feature:
            return values
        positions = np.linspace(0, len(values) - 1, self.config.max_values_per_feature, dtype=int)
        return values[positions]

    def _candidate_pairs(self, directed_features: list[str], candidates: list[str]) -> list[tuple[str, str]]:
        pairs: list[tuple[float, str, str]] = []
        for directed_feature in directed_features:
            for candidate in candidates:
                correlation = abs(
                    pd.Series(self.train_frame[directed_feature]).corr(
                        pd.Series(self.train_frame[candidate]), method="spearman"
                    )
                )
                correlation = 1.0 if np.isnan(correlation) else correlation
                score = (self.contributions[directed_feature] + self.contributions[candidate]) * (1.0 - correlation)
                pairs.append((score, directed_feature, candidate))
        pairs.sort(reverse=True)
        return [(left, right) for _, left, right in pairs[: self.config.max_pairs]]

    def _dynamic_threshold(self, left: str, right: str) -> float:
        left_score = self.contributions[left]
        right_score = self.contributions[right]
        similarity = 1.0 - abs(left_score - right_score) / (max(left_score, right_score) + 1e-12)
        return max(left_score, right_score) * (1.0 + self.config.response_margin * similarity)

    def _joint_response(self, left: str, right: str, contexts: pd.DataFrame) -> float:
        left_values = self._limited_values(left)
        right_values = self._limited_values(right)
        if len(left_values) == 0 or len(right_values) == 0 or len(contexts) == 0:
            return 0.0
        pairs = list(product(left_values, right_values))
        maximum = 0.0
        for _, context in contexts.iterrows():
            batches: list[pd.DataFrame] = []
            for start in range(0, len(pairs), self.config.inference_batch_size):
                group = pairs[start : start + self.config.inference_batch_size]
                batch = pd.DataFrame(
                    np.repeat(context.to_numpy()[None, :], len(group), axis=0),
                    columns=self.all_features,
                )
                batch[left] = [pair[0] for pair in group]
                batch[right] = [pair[1] for pair in group]
                batches.append(batch)
            baseline = self._distribution(context.to_frame().T)[0]
            for batch in batches:
                response = js_divergence(
                    self._distribution(batch),
                    np.repeat(baseline[None, :], len(batch), axis=0),
                )
                maximum = max(maximum, float(np.max(response)))
        return maximum

    def run(self, directed_features: list[str], high_contribution_features: list[str]) -> JointResponseResult:
        directed = list(dict.fromkeys(directed_features))
        candidates = [name for name in high_contribution_features if name not in set(directed)]
        if not directed or not candidates:
            return JointResponseResult([], directed, {})
        contexts = self._contexts()
        responses: dict[tuple[str, str], float] = {}
        added: list[str] = []
        for left, right in self._candidate_pairs(directed, candidates):
            if right in added:
                continue
            response = self._joint_response(left, right, contexts)
            responses[(left, right)] = response
            if response >= self._dynamic_threshold(left, right):
                added.append(right)
        return JointResponseResult(added, directed + added, responses)

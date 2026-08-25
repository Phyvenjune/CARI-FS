from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SplitConfig:
    seed: int = 42
    outer_test_size: float = 0.20
    validation_size: float = 0.20


@dataclass(frozen=True)
class ModelConfig:
    max_epochs: int = 150
    early_stopping_patience: int = 20
    reduce_lr_patience: int = 6
    batch_size_small: int = 32
    batch_size_medium: int = 128
    batch_size_large: int = 512
    l2_penalty: float = 0.005
    dropout: float = 0.35


@dataclass(frozen=True)
class ContributionConfig:
    diversity_high: float = 0.90
    diversity_low: float = 0.10
    high_contribution_method: str = "quantile"
    high_contribution_quantile: float = 0.75
    inference_batch_size: int = 2048


@dataclass(frozen=True)
class StructureConfig:
    dependency_threshold: float = 0.08
    max_condition_order: int = 2
    max_neighbors_for_conditioning: int = 6
    bootstrap_rounds: int = 20
    stability_threshold: float = 0.60
    max_path_length: int = 3
    orient_target_endpoints: bool = True


@dataclass(frozen=True)
class JointResponseConfig:
    max_pairs: int = 200
    max_values_per_feature: int = 32
    max_contexts: int = 8
    response_margin: float = 0.05
    inference_batch_size: int = 2048


@dataclass(frozen=True)
class ExperimentConfig:
    split: SplitConfig = SplitConfig()
    model: ModelConfig = ModelConfig()
    contribution: ContributionConfig = ContributionConfig()
    structure: StructureConfig = StructureConfig()
    joint_response: JointResponseConfig = JointResponseConfig()

from __future__ import annotations

import time
from dataclasses import asdict

import pandas as pd

from config import ExperimentConfig
from data_protocol import TabularProtocol
from metrics import evaluate_predictions, predict_classes
from model_factory import fit_with_internal_validation
from stage1_contribution import run_stage1
from stage2_structure_paths import run_stage2
from stage3_joint_response import JointResponseSupplementer


def run_cari_fs(frame: pd.DataFrame, target_name: str, config: ExperimentConfig | None = None) -> dict:
    """Run CARI-FS without using the outer test split during selection or early stopping."""
    config = config or ExperimentConfig()
    protocol = TabularProtocol(frame, target_name, config.split)
    split = protocol.split
    all_features = split.feature_names

    def all_feature_inputs(values: pd.DataFrame):
        return protocol.model_array(values, all_features)

    baseline_start = time.perf_counter()
    baseline_model = fit_with_internal_validation(
        all_feature_inputs(split.x_train),
        split.y_train,
        all_feature_inputs(split.x_val),
        split.y_val,
        protocol.class_count,
        config.split.seed,
        config.model,
    )
    baseline_time = time.perf_counter() - baseline_start

    phase1_start = time.perf_counter()
    phase1 = run_stage1(
        baseline_model,
        split.x_train,
        all_features,
        all_feature_inputs,
        config.contribution,
    )
    phase1_time = time.perf_counter() - phase1_start

    phase2_start = time.perf_counter()
    phase2 = run_stage2(
        split.x_train,
        split.y_train,
        phase1.high_contribution_features,
        phase1.contributions,
        config.structure,
        config.split.seed,
    )
    phase2_time = time.perf_counter() - phase2_start

    phase3_start = time.perf_counter()
    phase3 = JointResponseSupplementer(
        baseline_model,
        split.x_train,
        all_features,
        all_feature_inputs,
        phase1.contributions,
        phase1.perturbation_sets,
        config.joint_response,
        config.split.seed,
    ).run(phase2.stable_target_path_features, phase1.high_contribution_features)
    phase3_time = time.perf_counter() - phase3_start
    selected_features = phase3.final_features
    if not selected_features:
        raise RuntimeError("CARI-FS selected no feature; review the structure screening configuration.")

    def selected_inputs(values: pd.DataFrame):
        return protocol.model_array(values, selected_features)

    final_start = time.perf_counter()
    final_model = fit_with_internal_validation(
        selected_inputs(split.x_train),
        split.y_train,
        selected_inputs(split.x_val),
        split.y_val,
        protocol.class_count,
        config.split.seed,
        config.model,
    )
    final_training_time = time.perf_counter() - final_start

    # The outer test split is read here for the first and only time.
    baseline_metrics = evaluate_predictions(
        split.y_test,
        predict_classes(baseline_model, all_feature_inputs(split.x_test), protocol.class_count),
    )
    selected_metrics = evaluate_predictions(
        split.y_test,
        predict_classes(final_model, selected_inputs(split.x_test), protocol.class_count),
    )
    reduction_rate = 100.0 * (1.0 - len(selected_features) / len(all_features))
    return {
        "method": "CARI-FS",
        "protocol": {
            "outer_split": "80/20 train-test",
            "internal_split": "80/20 subtrain-validation within outer training",
            "early_stopping": "validation loss on internal validation split",
            "test_usage": "one final evaluation only",
            "weighted_f1": "one-versus-rest F1 weighted by test-set class support",
        },
        "dimensions": {
            "original_count": len(all_features),
            "selected_count": len(selected_features),
            "reduction_rate_percent": reduction_rate,
        },
        "stage1_feature_contribution_quantification": {
            "high_contribution_threshold": phase1.threshold,
            "high_contribution_features": phase1.high_contribution_features,
            "contributions": phase1.contributions,
        },
        "stage2_causality_informed_structure_path_screening": {
            "stable_target_local_directed_features": phase2.stable_target_path_features,
            "bootstrap_feature_stability": phase2.feature_stability,
            "directed_edges_last_bootstrap": phase2.directed_edges,
        },
        "stage3_joint_perturbation_response_supplement": {
            "added_features": phase3.added_features,
            "pair_responses": {f"{left}__{right}": score for (left, right), score in phase3.pair_responses.items()},
        },
        "metrics": {
            "all_features": baseline_metrics,
            "selected_features": selected_metrics,
            "accuracy_change_percentage_points": 100.0 * (selected_metrics["accuracy"] - baseline_metrics["accuracy"]),
            "weighted_f1_change_percentage_points": 100.0 * (
                selected_metrics["f1_weighted"] - baseline_metrics["f1_weighted"]
            ),
        },
        "execution_time_seconds": {
            "baseline_training": baseline_time,
            "stage1_feature_contribution_quantification": phase1_time,
            "stage2_structure_path_screening": phase2_time,
            "stage3_joint_response_supplement": phase3_time,
            "final_selected_model_training": final_training_time,
        },
        "selected_features": selected_features,
        "config": asdict(config),
    }

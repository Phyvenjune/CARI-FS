from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
import pandas as pd

from config import StructureConfig


@dataclass
class StructureResult:
    stable_target_path_features: list[str]
    feature_stability: dict[str, float]
    directed_edges: list[tuple[str, str]]


def _pair(left: str, right: str) -> tuple[str, str]:
    return tuple(sorted((left, right)))


def _residual(values: np.ndarray, conditions: np.ndarray | None) -> np.ndarray:
    if conditions is None or conditions.size == 0:
        return values - np.mean(values)
    design = np.column_stack([np.ones(len(values)), conditions])
    coefficients, *_ = np.linalg.lstsq(design, values, rcond=None)
    return values - design @ coefficients


def _weighted_dependency_score(
    frame: pd.DataFrame,
    left: str,
    right: str,
    conditions: list[str],
    contributions: dict[str, float],
) -> float:
    x = frame[left].to_numpy(dtype=float)
    y = frame[right].to_numpy(dtype=float)
    condition_values = frame[conditions].to_numpy(dtype=float) if conditions else None
    x_residual = _residual(x, condition_values)
    y_residual = _residual(y, condition_values)
    dependency = abs(pd.Series(x_residual).corr(pd.Series(y_residual), method="spearman"))
    dependency = 0.0 if np.isnan(dependency) else float(dependency)

    max_contribution = max(contributions.values(), default=1.0) or 1.0
    contribution_weight = 0.5 + 0.5 * (
        contributions.get(left, max_contribution) + contributions.get(right, max_contribution)
    ) / (2 * max_contribution)
    residual_noise = 0.5 * (
        np.std(x_residual) / (np.std(x) + 1e-12)
        + np.std(y_residual) / (np.std(y) + 1e-12)
    )
    noise_weight = 1.0 / (1.0 + residual_noise)
    sample_weight = min(1.0, np.sqrt(len(frame) / 200.0))
    return dependency * contribution_weight * noise_weight * sample_weight


def build_weighted_skeleton(
    frame: pd.DataFrame,
    nodes: list[str],
    contributions: dict[str, float],
    config: StructureConfig,
) -> tuple[set[tuple[str, str]], dict[tuple[str, str], set[str]], dict[tuple[str, str], float]]:
    edges = {_pair(left, right) for left, right in combinations(nodes, 2)}
    separating_sets: dict[tuple[str, str], set[str]] = {}
    scores: dict[tuple[str, str], float] = {}

    for order in range(config.max_condition_order + 1):
        for edge in list(edges):
            left, right = edge
            neighbors = sorted(
                ({node for candidate in edges for node in candidate if left in candidate and node != left}
                 | {node for candidate in edges for node in candidate if right in candidate and node != right})
                - {left, right}
            )[: config.max_neighbors_for_conditioning]
            condition_sets = [()] if order == 0 else combinations(neighbors, order)
            for condition_set in condition_sets:
                score = _weighted_dependency_score(frame, left, right, list(condition_set), contributions)
                scores[edge] = score
                if score < config.dependency_threshold:
                    edges.remove(edge)
                    separating_sets[edge] = set(condition_set)
                    break
    return edges, separating_sets, scores


def _adjacent(node_a: str, node_b: str, undirected: set[tuple[str, str]], directed: set[tuple[str, str]]) -> bool:
    return _pair(node_a, node_b) in undirected or (node_a, node_b) in directed or (node_b, node_a) in directed


def _orient(edge: tuple[str, str], source: str, target: str, undirected: set[tuple[str, str]], directed: set[tuple[str, str]]) -> bool:
    pair = _pair(*edge)
    if pair not in undirected:
        return False
    undirected.remove(pair)
    directed.add((source, target))
    return True


def orient_edges(
    nodes: list[str],
    skeleton: set[tuple[str, str]],
    separating_sets: dict[tuple[str, str], set[str]],
    target_name: str,
    config: StructureConfig,
) -> set[tuple[str, str]]:
    undirected = set(skeleton)
    directed: set[tuple[str, str]] = set()

    # Identify V-structures from non-adjacent endpoints and their separating sets.
    for middle in nodes:
        neighbors = [node for node in nodes if node != middle and _pair(node, middle) in undirected]
        for left, right in combinations(neighbors, 2):
            if not _adjacent(left, right, undirected, directed):
                separator = separating_sets.get(_pair(left, right), set())
                if middle not in separator:
                    _orient(_pair(left, middle), left, middle, undirected, directed)
                    _orient(_pair(right, middle), right, middle, undirected, directed)

    # Repeatedly apply the two Meek rules needed by this screening graph.
    changed = True
    while changed:
        changed = False
        for source, middle in list(directed):
            for target in nodes:
                if target in (source, middle):
                    continue
                if _pair(middle, target) in undirected and not _adjacent(source, target, undirected, directed):
                    changed = _orient(_pair(middle, target), middle, target, undirected, directed) or changed
        for left, right in list(undirected):
            has_directed_chain = any(
                (left, middle) in directed and (middle, right) in directed for middle in nodes
            )
            if has_directed_chain:
                changed = _orient((left, right), left, right, undirected, directed) or changed

    # Remaining target-adjacent links are oriented only as a target-local screening convention.
    if config.orient_target_endpoints:
        for left, right in list(undirected):
            if right == target_name:
                _orient((left, right), left, right, undirected, directed)
            elif left == target_name:
                _orient((left, right), right, left, undirected, directed)
    return directed


def target_path_features(
    directed_edges: set[tuple[str, str]], target_name: str, max_path_length: int
) -> set[str]:
    reverse_adjacency: dict[str, set[str]] = {}
    for source, target in directed_edges:
        reverse_adjacency.setdefault(target, set()).add(source)
    selected: set[str] = set()
    frontier = {target_name}
    for _ in range(max_path_length):
        previous = set().union(*(reverse_adjacency.get(node, set()) for node in frontier)) if frontier else set()
        previous -= selected
        selected.update(previous)
        frontier = previous
    selected.discard(target_name)
    return selected


def _single_structure_pass(
    frame: pd.DataFrame,
    candidate_features: list[str],
    target_name: str,
    contributions: dict[str, float],
    config: StructureConfig,
) -> tuple[set[str], set[tuple[str, str]]]:
    nodes = candidate_features + [target_name]
    skeleton, separating_sets, _ = build_weighted_skeleton(frame, nodes, contributions, config)
    directed = orient_edges(nodes, skeleton, separating_sets, target_name, config)
    return target_path_features(directed, target_name, config.max_path_length), directed


def run_stage2(
    train_frame: pd.DataFrame,
    train_labels: np.ndarray,
    candidate_features: list[str],
    contributions: dict[str, float],
    config: StructureConfig,
    seed: int,
) -> StructureResult:
    if not candidate_features:
        return StructureResult([], {}, [])
    target_name = "__target__"
    working = train_frame[candidate_features].copy()
    working[target_name] = train_labels
    bootstrap_counts = {feature: 0 for feature in candidate_features}
    final_edges: set[tuple[str, str]] = set()
    rng = np.random.default_rng(seed)
    for _ in range(config.bootstrap_rounds):
        indices = rng.integers(0, len(working), len(working))
        sample = working.iloc[indices].reset_index(drop=True)
        features, directed_edges = _single_structure_pass(
            sample, candidate_features, target_name, contributions, config
        )
        for feature in features:
            bootstrap_counts[feature] += 1
        final_edges = directed_edges
    stability = {feature: count / config.bootstrap_rounds for feature, count in bootstrap_counts.items()}
    stable = [feature for feature in candidate_features if stability[feature] >= config.stability_threshold]
    return StructureResult(stable, stability, sorted(final_edges))

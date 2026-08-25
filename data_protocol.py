from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, StandardScaler

from config import SplitConfig


@dataclass
class DataSplit:
    x_train: pd.DataFrame
    x_val: pd.DataFrame
    x_test: pd.DataFrame
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray
    feature_names: list[str]
    target_name: str


class TabularProtocol:
    """Owns the nested split and all train-only transformations."""

    def __init__(self, frame: pd.DataFrame, target_name: str, config: SplitConfig):
        if target_name not in frame.columns:
            raise ValueError(f"Missing target column: {target_name}")
        self.target_name = target_name
        self.feature_names = [name for name in frame.columns if name != target_name]
        self.config = config
        self._category_columns: list[str] = []
        self._numeric_columns: list[str] = []
        self._ordinal_encoder: OrdinalEncoder | None = None
        self._scaler: StandardScaler | None = None
        self._label_encoder = LabelEncoder()
        self._split = self._make_split(frame)

    @property
    def split(self) -> DataSplit:
        return self._split

    def _safe_stratify(self, y: pd.Series | np.ndarray) -> pd.Series | np.ndarray | None:
        values = pd.Series(y)
        return values if values.value_counts().min() >= 2 else None

    def _make_split(self, frame: pd.DataFrame) -> DataSplit:
        x_all = frame[self.feature_names].copy()
        y_all = frame[self.target_name].copy()
        x_outer_train, x_test, y_outer_train, y_test = train_test_split(
            x_all,
            y_all,
            test_size=self.config.outer_test_size,
            random_state=self.config.seed,
            stratify=self._safe_stratify(y_all),
        )
        x_train, x_val, y_train, y_val = train_test_split(
            x_outer_train,
            y_outer_train,
            test_size=self.config.validation_size,
            random_state=self.config.seed,
            stratify=self._safe_stratify(y_outer_train),
        )

        self._fit_transformers(x_train)
        x_train_encoded = self.encode(x_train)
        x_val_encoded = self.encode(x_val)
        x_test_encoded = self.encode(x_test)
        self._label_encoder.fit(y_train.astype(str))

        def encode_target(values: pd.Series) -> np.ndarray:
            return self._label_encoder.transform(values.astype(str))

        return DataSplit(
            x_train=x_train_encoded,
            x_val=x_val_encoded,
            x_test=x_test_encoded,
            y_train=encode_target(y_train),
            y_val=encode_target(y_val),
            y_test=encode_target(y_test),
            feature_names=self.feature_names,
            target_name=self.target_name,
        )

    def _fit_transformers(self, x_train: pd.DataFrame) -> None:
        self._category_columns = x_train.select_dtypes(exclude=[np.number]).columns.tolist()
        self._numeric_columns = [c for c in self.feature_names if c not in self._category_columns]
        if self._category_columns:
            self._ordinal_encoder = OrdinalEncoder(
                handle_unknown="use_encoded_value",
                unknown_value=-1,
                encoded_missing_value=-1,
            )
            self._ordinal_encoder.fit(x_train[self._category_columns].fillna("__missing__").astype(str))
        encoded = self._encode_without_scaling(x_train)
        self._scaler = StandardScaler().fit(encoded[self.feature_names])

    def _encode_without_scaling(self, frame: pd.DataFrame) -> pd.DataFrame:
        result = pd.DataFrame(index=frame.index)
        for column in self._numeric_columns:
            values = pd.to_numeric(frame[column], errors="coerce")
            fill_value = values.median()
            result[column] = values.fillna(0.0 if pd.isna(fill_value) else fill_value).astype(float)
        if self._category_columns:
            if self._ordinal_encoder is None:
                raise RuntimeError("Categorical encoder is not fitted")
            result[self._category_columns] = self._ordinal_encoder.transform(
                frame[self._category_columns].fillna("__missing__").astype(str)
            )
        return result[self.feature_names]

    def encode(self, frame: pd.DataFrame) -> pd.DataFrame:
        return self._encode_without_scaling(frame).reset_index(drop=True)

    def model_array(self, frame: pd.DataFrame, features: Iterable[str]) -> np.ndarray:
        if self._scaler is None:
            raise RuntimeError("Scaler is not fitted")
        names = list(features)
        scaled = self._scaler.transform(frame[self.feature_names])
        indices = [self.feature_names.index(name) for name in names]
        return scaled[:, indices].astype(np.float32)[..., np.newaxis]

    @property
    def class_count(self) -> int:
        return len(self._label_encoder.classes_)

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


@dataclass(frozen=True)
class TrainConfig:
    target_col: str
    test_size: float
    random_state: int
    model_name: str  # "LogReg" | "RandomForest" | "GradientBoosting"


def build_preprocessor(numeric_cols: list[str], categorical_cols: list[str]) -> ColumnTransformer:
    num_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    cat_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", num_pipe, numeric_cols),
            ("cat", cat_pipe, categorical_cols),
        ],
        remainder="drop",
    )


def get_model(model_name: str):
    if model_name == "LogReg":
        return LogisticRegression(max_iter=2000)
    if model_name == "RandomForest":
        return RandomForestClassifier(n_estimators=300, random_state=42)
    if model_name == "GradientBoosting":
        return GradientBoostingClassifier(random_state=42)
    raise ValueError(f"Unknown model: {model_name}")


def train_and_evaluate(
    df: pd.DataFrame,
    *,
    numeric_cols: list[str],
    categorical_cols: list[str],
    cfg: TrainConfig,
) -> tuple[Pipeline, dict[str, Any]]:
    if cfg.target_col not in df.columns:
        raise ValueError(f"Target column '{cfg.target_col}' not found.")

    X = df.drop(columns=[cfg.target_col])
    y_raw = df[cfg.target_col]

    # map to binary like the notebook (but keep original names for display)
    if y_raw.dtype == object or str(y_raw.dtype).startswith("category"):
        y = y_raw.map({"Risque Elevé": 0, "Risque Faible": 1})
        if y.isna().any():
            # fallback: factorize
            y, uniques = pd.factorize(y_raw)
            target_names = [str(u) for u in uniques]
        else:
            target_names = ["Risque Elevé", "Risque Faible"]
    else:
        y = y_raw
        target_names = sorted(pd.Series(y_raw).dropna().unique().tolist())

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=cfg.test_size, random_state=cfg.random_state, stratify=y if len(pd.unique(y)) > 1 else None
    )

    pre = build_preprocessor(numeric_cols=numeric_cols, categorical_cols=categorical_cols)
    clf = get_model(cfg.model_name)
    pipe = Pipeline(steps=[("pre", pre), ("clf", clf)])
    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)

    metrics: dict[str, Any] = {
        "model": cfg.model_name,
        "test_size": cfg.test_size,
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "classification_report": classification_report(y_test, y_pred, zero_division=0),
        "target_names": target_names,
        "y_test_values": list(map(int, y_test.tolist())) if hasattr(y_test, "tolist") else list(y_test),
    }

    # AUC and test probabilities if probabilities exist and binary target
    if hasattr(pipe, "predict_proba"):
        try:
            proba = pipe.predict_proba(X_test)[:, 1]
            metrics["roc_auc"] = float(roc_auc_score(y_test, proba))
            metrics["y_test_proba"] = [float(x) for x in proba.tolist()]
        except Exception:
            pass

    # training ranges for sanity checks on user inputs
    ranges = {}
    for c in numeric_cols:
        s = pd.to_numeric(df[c], errors="coerce")
        ranges[c] = {"min": float(np.nanmin(s)), "max": float(np.nanmax(s))}
    metrics["numeric_ranges"] = ranges

    return pipe, metrics


def save_artifact(path: Path, *, pipeline: Pipeline, metadata: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": pipeline, "metadata": metadata}, path)


def load_artifact(path: Path) -> dict[str, Any]:
    obj = joblib.load(path)
    if not isinstance(obj, dict) or "pipeline" not in obj:
        raise ValueError("Invalid artifact file.")
    return obj



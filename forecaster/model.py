"""Direction classifier: a soft-voting ensemble validated walk-forward."""

from __future__ import annotations

import os
from dataclasses import dataclass, field

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEED = 42


def build_ensemble() -> VotingClassifier:
    boosting = HistGradientBoostingClassifier(
        max_depth=3,
        learning_rate=0.08,
        max_iter=120,
        min_samples_leaf=40,
        l2_regularization=1.0,
        random_state=SEED,
    )
    forest = make_pipeline(
        SimpleImputer(strategy="median"),
        RandomForestClassifier(
            n_estimators=160,
            max_depth=5,
            min_samples_leaf=25,
            random_state=SEED,
            n_jobs=1,
        ),
    )
    logistic = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(max_iter=400, C=0.3, random_state=SEED),
    )
    return VotingClassifier(
        estimators=[
            ("gradient_boosting", boosting),
            ("random_forest", forest),
            ("logistic", logistic),
        ],
        voting="soft",
        n_jobs=1,
    )


def _unstick(frame: pd.DataFrame) -> pd.DataFrame:
    """Give a constant column two values so histogram boosting can bin it.

    A one-row nudge cannot form its own leaf (`min_samples_leaf` is 40), so the
    trees ignore it. Real Nifty features are rarely constant; a short synthetic
    window can be.
    """
    out = frame.copy()
    for column in out.columns:
        if out[column].nunique(dropna=True) >= 2:
            continue
        position = out.columns.get_loc(column)
        current = out.iloc[0, position]
        out.iloc[0, position] = (0.0 if pd.isna(current) else float(current)) + 1e-6
    return out


@dataclass
class ForecastModel:
    horizon: int
    features: list[str]
    model: VotingClassifier
    oos: pd.DataFrame
    metrics: dict = field(default_factory=dict)
    importance: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))

    def predict_proba(self, row: pd.DataFrame) -> float:
        frame = row.loc[:, self.features]
        probabilities = self.model.predict_proba(frame)[:, 1]
        return float(probabilities[0])

    def component_probas(self, row: pd.DataFrame) -> dict[str, float]:
        frame = row.loc[:, self.features]
        return {
            name: float(estimator.predict_proba(frame)[:, 1][0])
            for name, estimator in self.model.named_estimators_.items()
        }


def _usable(features: pd.DataFrame, target: pd.Series) -> pd.Series:
    mask = target.notna()
    coverage = features.notna().mean(axis=1)
    return mask & coverage.gt(0.6)


def train(
    features: pd.DataFrame,
    target: pd.Series,
    forward_return: pd.Series,
    horizon: int,
) -> ForecastModel:
    mask = _usable(features, target)
    usable = features.loc[mask]
    y = target.loc[mask].astype(int)
    returns = forward_return.reindex(usable.index)
    if len(usable) < 300:
        raise ValueError(f"Only {len(usable)} usable rows; need at least 300 to train.")

    columns = list(usable.columns)
    splits = 5 if len(usable) >= 800 else 3
    splitter = TimeSeriesSplit(n_splits=splits, gap=horizon)
    pieces: list[pd.DataFrame] = []
    for train_idx, test_idx in splitter.split(usable):
        ensemble = build_ensemble()
        ensemble.fit(_unstick(usable.iloc[train_idx]), y.iloc[train_idx])
        proba = ensemble.predict_proba(usable.iloc[test_idx])[:, 1]
        pieces.append(
            pd.DataFrame(
                {
                    "proba": proba,
                    "predicted": (proba >= 0.5).astype(int),
                    "actual": y.iloc[test_idx].to_numpy(),
                    "forward_return": returns.iloc[test_idx].to_numpy(),
                },
                index=usable.index[test_idx],
            )
        )
    oos = pd.concat(pieces)
    oos = oos[~oos.index.duplicated(keep="last")].sort_index()

    model = build_ensemble()
    model.fit(_unstick(usable), y)
    forest = model.named_estimators_["random_forest"].named_steps["randomforestclassifier"]
    importance = pd.Series(forest.feature_importances_, index=columns).sort_values(ascending=False)
    trained = ForecastModel(horizon=horizon, features=columns, model=model, oos=oos, importance=importance)
    trained.metrics = evaluate(oos)
    return trained


def evaluate(oos: pd.DataFrame) -> dict:
    actual = oos["actual"].fillna(0).astype(int)
    proba = oos["proba"].astype(float)
    predicted = oos["predicted"].fillna(0).astype(int)
    base_rate = float(actual.mean())
    metrics = {
        "rows": int(len(oos)),
        "accuracy": float(accuracy_score(actual, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(actual, predicted)),
        "base_rate": base_rate,
        "base_accuracy": float(max(base_rate, 1 - base_rate)),
        "brier": float(np.mean((proba - actual) ** 2)),
        "base_brier": float(base_rate * (1 - base_rate)),
    }
    if actual.nunique() > 1:
        metrics["roc_auc"] = float(roc_auc_score(actual, proba))
    else:
        metrics["roc_auc"] = float("nan")
    return metrics


def equity_curves(oos: pd.DataFrame) -> pd.DataFrame:
    """Growth of 1 rupee. Horizon-1 only: overlapping multi-day returns are not a path."""
    returns = oos["forward_return"].fillna(0.0)
    position = oos["predicted"].fillna(0).astype(float)
    curves = pd.DataFrame(
        {
            "Model (long/flat)": (1 + returns * position).cumprod(),
            "Buy and hold": (1 + returns).cumprod(),
        },
        index=oos.index,
    )
    return curves

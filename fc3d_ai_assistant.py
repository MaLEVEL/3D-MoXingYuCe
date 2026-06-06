#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
福彩3D模拟分析助手：单文件入门版

本文件把数据读取、特征工程、统计模型、机器学习模型、可选 PyTorch
TabResNet、集成、回测和候选号码输出放在一个文件里，方便初学者直接运行。

重要安全提示：
本工具仅用于学习和模拟研究，不代表真实预测，不建议用于投注。
"""

from __future__ import annotations

import argparse
import math
import sys
import warnings
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False


SAFETY_TEXT = "本工具仅用于学习和模拟研究，不代表真实预测，不建议用于投注。"
SIMULATION_TEXT = "仅为模型模拟，不代表真实开奖结果。"
RANDOM_BASELINE_TEXT = "如果模型没有显著超过随机基线，说明历史数据中没有稳定可学习规律。"

MODEL_SCORE_CALIBRATION_EXPONENT = 0.72
MODEL_SCORE_UNIFORM_BLEND = 0.08
MIN_TORCH_TRAIN_SAMPLES = 1000

CODES = np.array([f"{i:03d}" for i in range(1000)])
CODE_DIGITS = np.array([[i // 100, (i // 10) % 10, i % 10] for i in range(1000)], dtype=int)
CODE_SUMS = CODE_DIGITS.sum(axis=1)
CODE_SPANS = CODE_DIGITS.max(axis=1) - CODE_DIGITS.min(axis=1)
SUM_COMBO_COUNTS = np.bincount(CODE_SUMS, minlength=28).astype(float)


def print_section(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def print_safety() -> None:
    print_section("安全提示")
    print(SAFETY_TEXT)
    print("所有候选号码都只能理解为数学建模输出，不能理解为购彩建议。")


def normalize_scores(scores: np.ndarray) -> np.ndarray:
    arr = np.asarray(scores, dtype=float).copy()
    arr[~np.isfinite(arr)] = 0.0
    min_value = arr.min(initial=0.0)
    if min_value < 0:
        arr = arr - min_value
    total = float(arr.sum())
    if total <= 0:
        return np.ones_like(arr, dtype=float) / len(arr)
    return arr / total


def calibrate_model_scores(
    scores: np.ndarray,
    exponent: float = MODEL_SCORE_CALIBRATION_EXPONENT,
    uniform_blend: float = MODEL_SCORE_UNIFORM_BLEND,
) -> np.ndarray:
    """Convert one model output to a softened relative-score scale, not a probability."""
    calibrated = normalize_scores(scores)
    if len(calibrated) == 0:
        return calibrated
    exponent = min(1.0, max(0.2, float(exponent)))
    uniform_blend = min(0.5, max(0.0, float(uniform_blend)))
    calibrated = np.power(calibrated + 1e-12, exponent)
    calibrated = normalize_scores(calibrated)
    uniform = np.ones_like(calibrated, dtype=float) / len(calibrated)
    calibrated = (1.0 - uniform_blend) * calibrated + uniform_blend * uniform
    return normalize_scores(calibrated)


def code_type_from_digits(digits: np.ndarray | list[int] | tuple[int, int, int]) -> str:
    unique_count = len(set(int(x) for x in digits))
    if unique_count == 1:
        return "豹子"
    if unique_count == 2:
        return "组三"
    return "组六"


def describe_code(code: str) -> dict[str, object]:
    digits = np.array([int(code[0]), int(code[1]), int(code[2])], dtype=int)
    return {
        "number": code,
        "sum": int(digits.sum()),
        "span": int(digits.max() - digits.min()),
        "type": code_type_from_digits(digits),
    }


def add_number_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["bai"] = out["number"].str[0].astype(int)
    out["shi"] = out["number"].str[1].astype(int)
    out["ge"] = out["number"].str[2].astype(int)
    digits = out[["bai", "shi", "ge"]].to_numpy()
    out["sum3"] = digits.sum(axis=1)
    out["span"] = digits.max(axis=1) - digits.min(axis=1)
    out["max_digit"] = digits.max(axis=1)
    out["min_digit"] = digits.min(axis=1)
    out["type"] = [code_type_from_digits(row) for row in digits]
    out["is_baozi"] = (out["type"] == "豹子").astype(int)
    out["is_zusan"] = (out["type"] == "组三").astype(int)
    out["is_zuliu"] = (out["type"] == "组六").astype(int)
    return out


def generate_sample_data(csv_path: Path, rows: int = 240, seed: int = 42, overwrite: bool = False) -> None:
    if csv_path.exists() and not overwrite:
        print(f"示例数据文件已存在，未覆盖：{csv_path}")
        return

    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=rows, freq="D")
    records = []
    for idx, date in enumerate(dates, start=1):
        digits = rng.integers(0, 10, size=3)
        records.append(
            {
                "issue": f"2024{idx:03d}",
                "date": date.strftime("%Y-%m-%d"),
                "number": "".join(str(int(x)) for x in digits),
            }
        )
    sample_df = pd.DataFrame(records)
    sample_df.to_csv(csv_path, index=False, encoding="utf-8")
    print(f"已生成可运行示例数据：{csv_path}")


def load_history_data(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"找不到 CSV 文件：{csv_path}")

    df = pd.read_csv(
        csv_path,
        dtype={"issue": "string", "date": "string", "number": "string"},
        keep_default_na=False,
    )

    required_cols = {"issue", "date", "number"}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(f"CSV 缺少必要字段：{sorted(missing_cols)}")

    df = df[["issue", "date", "number"]].copy()
    for col in ["issue", "date", "number"]:
        df[col] = df[col].astype("string").str.strip()
        df.loc[df[col] == "", col] = pd.NA

    problems: list[str] = []
    missing_rows = df[df[["issue", "date", "number"]].isna().any(axis=1)]
    if not missing_rows.empty:
        problems.append(f"存在缺失值行：{missing_rows.index.tolist()[:10]}")

    duplicate_rows = df[df.duplicated("issue", keep=False)]
    if not duplicate_rows.empty:
        examples = duplicate_rows["issue"].dropna().astype(str).unique().tolist()[:10]
        problems.append(f"存在重复期号：{examples}")

    invalid_number = df["number"].isna() | ~df["number"].str.fullmatch(r"\d{3}").fillna(False)
    if invalid_number.any():
        examples = df.loc[invalid_number, ["issue", "number"]].head(10).to_dict("records")
        problems.append(f"存在异常号码，号码必须是 000-999 的三位字符串：{examples}")

    parsed_dates = pd.to_datetime(df["date"], errors="coerce")
    if parsed_dates.isna().any():
        examples = df.loc[parsed_dates.isna(), ["issue", "date"]].head(10).to_dict("records")
        problems.append(f"存在异常日期，建议格式 YYYY-MM-DD：{examples}")

    if problems:
        message = "\n".join(f"- {item}" for item in problems)
        raise ValueError(f"数据校验失败：\n{message}")

    df["_date"] = parsed_dates
    df = df.sort_values(["_date", "issue"]).drop(columns=["_date"]).reset_index(drop=True)
    return add_number_columns(df)


def build_history_feature(history: pd.DataFrame, windows: tuple[int, ...] = (30, 60, 120)) -> dict[str, float]:
    features: dict[str, float] = {}
    n = len(history)
    features["history_len"] = float(n)

    if n == 0:
        features.update(
            {
                "prev_bai": 0.0,
                "prev_shi": 0.0,
                "prev_ge": 0.0,
                "prev_sum3": 0.0,
                "prev_span": 0.0,
                "prev_is_baozi": 0.0,
                "prev_is_zusan": 0.0,
                "prev_is_zuliu": 0.0,
            }
        )
    else:
        prev = history.iloc[-1]
        features.update(
            {
                "prev_bai": float(prev["bai"]),
                "prev_shi": float(prev["shi"]),
                "prev_ge": float(prev["ge"]),
                "prev_sum3": float(prev["sum3"]),
                "prev_span": float(prev["span"]),
                "prev_is_baozi": float(prev["is_baozi"]),
                "prev_is_zusan": float(prev["is_zusan"]),
                "prev_is_zuliu": float(prev["is_zuliu"]),
            }
        )

    alpha = 1.0
    if n == 0:
        all_counts = np.zeros(10, dtype=float)
        pos_counts = {pos: np.zeros(10, dtype=float) for pos in ["bai", "shi", "ge"]}
        sum_counts = np.zeros(28, dtype=float)
        span_counts = np.zeros(10, dtype=float)
        type_counts = Counter()
    else:
        all_digits = history[["bai", "shi", "ge"]].to_numpy(dtype=int).ravel()
        all_counts = np.bincount(all_digits, minlength=10).astype(float)
        pos_counts = {
            pos: np.bincount(history[pos].to_numpy(dtype=int), minlength=10).astype(float)
            for pos in ["bai", "shi", "ge"]
        }
        sum_counts = np.bincount(history["sum3"].to_numpy(dtype=int), minlength=28).astype(float)
        span_counts = np.bincount(history["span"].to_numpy(dtype=int), minlength=10).astype(float)
        type_counts = Counter(history["type"].astype(str))

    for digit in range(10):
        features[f"hist_digit_freq_{digit}"] = float((all_counts[digit] + alpha) / (3 * n + 10 * alpha))
        for pos in ["bai", "shi", "ge"]:
            features[f"hist_{pos}_freq_{digit}"] = float((pos_counts[pos][digit] + alpha) / (n + 10 * alpha))

    for sum_value in range(28):
        features[f"hist_sum_freq_{sum_value}"] = float((sum_counts[sum_value] + alpha) / (n + 28 * alpha))
    for span_value in range(10):
        features[f"hist_span_freq_{span_value}"] = float((span_counts[span_value] + alpha) / (n + 10 * alpha))
    for type_name in ["豹子", "组三", "组六"]:
        features[f"hist_type_freq_{type_name}"] = float((type_counts[type_name] + alpha) / (n + 3 * alpha))

    for window in windows:
        wdf = history.tail(window)
        m = len(wdf)
        prefix = f"roll{window}"
        if m == 0:
            features[f"{prefix}_sum_mean"] = 0.0
            features[f"{prefix}_sum_std"] = 0.0
            features[f"{prefix}_span_mean"] = 0.0
            roll_all_counts = np.zeros(10, dtype=float)
            roll_pos_counts = {pos: np.zeros(10, dtype=float) for pos in ["bai", "shi", "ge"]}
            roll_type_counts = Counter()
        else:
            features[f"{prefix}_sum_mean"] = float(wdf["sum3"].mean())
            features[f"{prefix}_sum_std"] = float(wdf["sum3"].std(ddof=0))
            features[f"{prefix}_span_mean"] = float(wdf["span"].mean())
            roll_digits = wdf[["bai", "shi", "ge"]].to_numpy(dtype=int).ravel()
            roll_all_counts = np.bincount(roll_digits, minlength=10).astype(float)
            roll_pos_counts = {
                pos: np.bincount(wdf[pos].to_numpy(dtype=int), minlength=10).astype(float)
                for pos in ["bai", "shi", "ge"]
            }
            roll_type_counts = Counter(wdf["type"].astype(str))

        for digit in range(10):
            features[f"{prefix}_digit_freq_{digit}"] = float((roll_all_counts[digit] + alpha) / (3 * m + 10 * alpha))
            for pos in ["bai", "shi", "ge"]:
                features[f"{prefix}_{pos}_freq_{digit}"] = float((roll_pos_counts[pos][digit] + alpha) / (m + 10 * alpha))
        for type_name in ["豹子", "组三", "组六"]:
            features[f"{prefix}_type_count_{type_name}"] = float(roll_type_counts[type_name])

    decay_pos_counts = {pos: np.zeros(10, dtype=float) for pos in ["bai", "shi", "ge"]}
    decay_total = 0.0
    if n:
        half_life = 45.0
        recent_rows = list(history[["bai", "shi", "ge"]].itertuples(index=False, name=None))
        for age, row in enumerate(reversed(recent_rows)):
            weight = math.exp(-age / half_life)
            decay_total += weight
            for pos, digit in zip(["bai", "shi", "ge"], row):
                decay_pos_counts[pos][int(digit)] += weight
    for digit in range(10):
        for pos in ["bai", "shi", "ge"]:
            features[f"expdecay_{pos}_freq_{digit}"] = float(
                (decay_pos_counts[pos][digit] + alpha) / (decay_total + 10 * alpha)
            )

    last_seen_digit = {digit: -1 for digit in range(10)}
    last_seen_pos = {(pos, digit): -1 for pos in ["bai", "shi", "ge"] for digit in range(10)}
    last_seen_sum = {value: -1 for value in range(28)}
    last_seen_span = {value: -1 for value in range(10)}
    last_seen_type = {value: -1 for value in ["豹子", "组三", "组六"]}

    for idx, row in enumerate(history.itertuples(index=False)):
        digits = [int(row.bai), int(row.shi), int(row.ge)]
        for digit in set(digits):
            last_seen_digit[digit] = idx
        for pos, digit in zip(["bai", "shi", "ge"], digits):
            last_seen_pos[(pos, digit)] = idx
        last_seen_sum[int(row.sum3)] = idx
        last_seen_span[int(row.span)] = idx
        last_seen_type[str(row.type)] = idx

    for digit in range(10):
        features[f"omit_digit_{digit}"] = float(n - last_seen_digit[digit] if last_seen_digit[digit] >= 0 else n + 1)
        for pos in ["bai", "shi", "ge"]:
            key = (pos, digit)
            current_omit = float(n - last_seen_pos[key] if last_seen_pos[key] >= 0 else n + 1)
            pos_values = history[pos].to_numpy(dtype=int) if n else np.array([], dtype=int)
            occurrence_indices = [idx for idx, value in enumerate(pos_values) if int(value) == digit]
            if occurrence_indices:
                gaps = np.diff(np.array([-1, *occurrence_indices, n], dtype=float))
                avg_omit = float(np.mean(gaps))
                max_omit = float(np.max(gaps))
            else:
                avg_omit = float(n + 1)
                max_omit = float(n + 1)
            hist_freq = float(features.get(f"hist_{pos}_freq_{digit}", 0.1))
            recent_freq = float(features.get(f"roll30_{pos}_freq_{digit}", hist_freq))
            heat = (recent_freq - hist_freq) / max(hist_freq, 1e-6)
            features[f"omit_{pos}_{digit}"] = current_omit
            features[f"avg_omit_{pos}_{digit}"] = avg_omit
            features[f"max_omit_{pos}_{digit}"] = max_omit
            features[f"heat_{pos}_{digit}"] = float(heat)
    for value in range(28):
        features[f"omit_sum_{value}"] = float(n - last_seen_sum[value] if last_seen_sum[value] >= 0 else n + 1)
    for value in range(10):
        features[f"omit_span_{value}"] = float(n - last_seen_span[value] if last_seen_span[value] >= 0 else n + 1)
    for type_name in ["豹子", "组三", "组六"]:
        features[f"omit_type_{type_name}"] = float(
            n - last_seen_type[type_name] if last_seen_type[type_name] >= 0 else n + 1
        )

    return features


def build_feature_table(df: pd.DataFrame, windows: tuple[int, ...] = (30, 60, 120)) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for idx in range(len(df)):
        row = build_history_feature(df.iloc[:idx], windows=windows)
        current = df.iloc[idx]
        row.update(
            {
                "row_index": idx,
                "target_number": str(current["number"]),
                "target_bai": int(current["bai"]),
                "target_shi": int(current["shi"]),
                "target_ge": int(current["ge"]),
            }
        )
        rows.append(row)
    feature_df = pd.DataFrame(rows).fillna(0.0)
    if len(feature_df) > 1:
        feature_df = feature_df.iloc[1:].reset_index(drop=True)
    return feature_df


def get_feature_columns(feature_df: pd.DataFrame) -> list[str]:
    excluded = {"row_index", "target_number", "target_bai", "target_shi", "target_ge"}
    return [col for col in feature_df.columns if col not in excluded]


def score_statistical_models(history: pd.DataFrame, seed: int = 42, alpha: float = 1.0) -> dict[str, np.ndarray]:
    history = add_number_columns(history) if "bai" not in history.columns else history
    n = len(history)
    rng = np.random.default_rng(seed + n)

    scores: dict[str, np.ndarray] = {}
    scores["random_baseline"] = normalize_scores(rng.random(len(CODES)))

    code_counts = Counter(history["number"].astype(str)) if n else Counter()
    raw_counts = np.array([code_counts[code] for code in CODES], dtype=float)
    scores["history_frequency"] = normalize_scores((raw_counts + 0.1) / (n + 1000 * 0.1))
    scores["bayes_smooth_frequency"] = normalize_scores((raw_counts + alpha) / (n + 1000 * alpha))

    last_seen_code: dict[str, int] = {}
    for idx, code in enumerate(history["number"].astype(str)):
        last_seen_code[code] = idx
    omissions = np.array([n - last_seen_code.get(code, -1) if code in last_seen_code else n + 1 for code in CODES], dtype=float)
    scores["omission"] = normalize_scores(np.log1p(omissions))

    if n:
        sum_counts = np.bincount(history["sum3"].to_numpy(dtype=int), minlength=28).astype(float)
        pos_probs = []
        for pos in ["bai", "shi", "ge"]:
            counts = np.bincount(history[pos].to_numpy(dtype=int), minlength=10).astype(float)
            pos_probs.append((counts + alpha) / (n + 10 * alpha))
    else:
        sum_counts = np.zeros(28, dtype=float)
        pos_probs = [np.ones(10, dtype=float) / 10 for _ in range(3)]

    sum_probs = (sum_counts + alpha) / (n + 28 * alpha)
    sum_scores = sum_probs[CODE_SUMS] / SUM_COMBO_COUNTS[CODE_SUMS]
    scores["sum_distribution"] = normalize_scores(sum_scores)

    position_scores = (
        pos_probs[0][CODE_DIGITS[:, 0]]
        * pos_probs[1][CODE_DIGITS[:, 1]]
        * pos_probs[2][CODE_DIGITS[:, 2]]
    )
    scores["position_frequency"] = normalize_scores(position_scores)
    return scores


def code_scores_from_position_probs(position_probs: tuple[np.ndarray, np.ndarray, np.ndarray]) -> np.ndarray:
    bai_prob, shi_prob, ge_prob = position_probs
    scores = bai_prob[CODE_DIGITS[:, 0]] * shi_prob[CODE_DIGITS[:, 1]] * ge_prob[CODE_DIGITS[:, 2]]
    return normalize_scores(scores)


def position_marginals_from_code_scores(scores: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    scores = normalize_scores(scores)
    marginals: list[np.ndarray] = []
    for pos in range(3):
        arr = np.zeros(10, dtype=float)
        for digit in range(10):
            arr[digit] = scores[CODE_DIGITS[:, pos] == digit].sum()
        marginals.append(normalize_scores(arr))
    return marginals[0], marginals[1], marginals[2]


@dataclass
class DigitModelBundle:
    name: str
    estimators: list[object | None]
    fallback_probs: list[np.ndarray]

    def predict_position_probs(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        all_position_probs: list[np.ndarray] = []
        for estimator, fallback in zip(self.estimators, self.fallback_probs):
            if estimator is None:
                probs = np.tile(fallback, (len(x), 1))
            else:
                raw = estimator.predict_proba(x)
                classes = getattr(estimator, "classes_", np.arange(raw.shape[1]))
                probs = np.zeros((len(x), 10), dtype=float)
                for raw_idx, class_value in enumerate(classes):
                    class_int = int(class_value)
                    if 0 <= class_int <= 9:
                        probs[:, class_int] = raw[:, raw_idx]
                probs = probs + 1e-6
                probs = probs / probs.sum(axis=1, keepdims=True)
            all_position_probs.append(probs)
        return all_position_probs[0], all_position_probs[1], all_position_probs[2]


def smoothed_digit_prior(y: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    counts = np.bincount(y.astype(int), minlength=10).astype(float)
    return normalize_scores(counts + alpha)


def fit_digit_bundle(
    name: str,
    estimator_factory: Callable[[], object],
    x_train: np.ndarray,
    y_train: np.ndarray,
) -> DigitModelBundle | None:
    estimators: list[object | None] = []
    fallback_probs: list[np.ndarray] = []
    for pos in range(3):
        y = y_train[:, pos].astype(int)
        fallback = smoothed_digit_prior(y)
        fallback_probs.append(fallback)
        if len(np.unique(y)) < 2 or len(y) < 20:
            estimators.append(None)
            continue
        try:
            estimator = estimator_factory()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                estimator.fit(x_train, y)
            estimators.append(estimator)
        except Exception as exc:
            print(f"模型 {name} 的第 {pos + 1} 位训练失败，已回退到平滑频率：{exc}")
            estimators.append(None)

    if all(est is None for est in estimators):
        return None
    return DigitModelBundle(name=name, estimators=estimators, fallback_probs=fallback_probs)


def optional_boosting_factories(seed: int) -> dict[str, Callable[[], object]]:
    factories: dict[str, Callable[[], object]] = {}

    try:
        from xgboost import XGBClassifier

        factories["xgboost"] = lambda: XGBClassifier(
            objective="multi:softprob",
            num_class=10,
            n_estimators=60,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="mlogloss",
            random_state=seed,
            verbosity=0,
        )
    except Exception:
        pass

    try:
        from lightgbm import LGBMClassifier

        factories["lightgbm"] = lambda: LGBMClassifier(
            objective="multiclass",
            num_class=10,
            n_estimators=80,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=seed,
            verbose=-1,
        )
    except Exception:
        pass

    try:
        from catboost import CatBoostClassifier

        factories["catboost"] = lambda: CatBoostClassifier(
            loss_function="MultiClass",
            iterations=80,
            depth=4,
            learning_rate=0.05,
            random_seed=seed,
            verbose=False,
        )
    except Exception:
        pass

    return factories


def fit_ml_models(
    x_train: np.ndarray,
    y_train: np.ndarray,
    seed: int = 42,
    enable_boosting: bool = False,
) -> dict[str, DigitModelBundle]:
    if not SKLEARN_AVAILABLE:
        print("未检测到 scikit-learn，已跳过传统机器学习模型。")
        return {}

    factories: dict[str, Callable[[], object]] = {
        "logistic_regression": lambda: Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=1000, solver="lbfgs", C=0.8)),
            ]
        ),
        "random_forest": lambda: RandomForestClassifier(
            n_estimators=160,
            max_depth=8,
            min_samples_leaf=2,
            random_state=seed,
            n_jobs=-1,
        ),
    }

    if enable_boosting:
        factories.update(optional_boosting_factories(seed))

    bundles: dict[str, DigitModelBundle] = {}
    for name, factory in factories.items():
        bundle = fit_digit_bundle(name, factory, x_train, y_train)
        if bundle is not None:
            bundles[name] = bundle
    return bundles


if TORCH_AVAILABLE:

    class ResidualBlock(nn.Module):
        def __init__(self, hidden_dim: int, dropout: float) -> None:
            super().__init__()
            self.block = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.Dropout(dropout),
            )
            self.norm = nn.LayerNorm(hidden_dim)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.norm(x + self.block(x))


    class TabResNet(nn.Module):
        def __init__(self, input_dim: int, hidden_dim: int = 128, dropout: float = 0.15, blocks: int = 3) -> None:
            super().__init__()
            self.input = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            self.body = nn.Sequential(*[ResidualBlock(hidden_dim, dropout) for _ in range(blocks)])
            self.head_bai = nn.Linear(hidden_dim, 10)
            self.head_shi = nn.Linear(hidden_dim, 10)
            self.head_ge = nn.Linear(hidden_dim, 10)

        def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            hidden = self.body(self.input(x))
            return self.head_bai(hidden), self.head_shi(hidden), self.head_ge(hidden)


@dataclass
class TorchConfig:
    enabled: bool = False
    epochs: int = 12
    batch_size: int = 32
    learning_rate: float = 1e-3
    hidden_dim: int = 128
    dropout: float = 0.15
    weight_decay: float = 1e-4


def fit_torch_tabresnet(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_predict: np.ndarray,
    config: TorchConfig,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    if not config.enabled:
        return None
    if not TORCH_AVAILABLE:
        print("未检测到 PyTorch，已跳过 TabResNet。")
        return None
    if len(x_train) < MIN_TORCH_TRAIN_SAMPLES:
        print(f"训练样本少于 {MIN_TORCH_TRAIN_SAMPLES}，已跳过 TabResNet，避免深度模型过拟合。")
        return None

    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mean = x_train.mean(axis=0, keepdims=True)
    std = x_train.std(axis=0, keepdims=True)
    std[std < 1e-6] = 1.0
    x_train_scaled = ((x_train - mean) / std).astype("float32")
    x_predict_scaled = ((x_predict - mean) / std).astype("float32")

    dataset = TensorDataset(
        torch.tensor(x_train_scaled, dtype=torch.float32),
        torch.tensor(y_train[:, 0], dtype=torch.long),
        torch.tensor(y_train[:, 1], dtype=torch.long),
        torch.tensor(y_train[:, 2], dtype=torch.long),
    )
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

    model = TabResNet(
        input_dim=x_train.shape[1],
        hidden_dim=config.hidden_dim,
        dropout=config.dropout,
        blocks=3,
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    model.train()
    for _ in range(config.epochs):
        for xb, yb, ys, yg in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            ys = ys.to(device)
            yg = yg.to(device)
            optimizer.zero_grad()
            logits_b, logits_s, logits_g = model(xb)
            loss = criterion(logits_b, yb) + criterion(logits_s, ys) + criterion(logits_g, yg)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        xp = torch.tensor(x_predict_scaled, dtype=torch.float32).to(device)
        logits = model(xp)
        probs = tuple(torch.softmax(item, dim=1).cpu().numpy() for item in logits)
    return probs


def evaluate_score_rows(actual_numbers: list[str], score_rows: list[np.ndarray]) -> dict[str, float]:
    if not score_rows:
        return {
            "samples": 0,
            "bai_acc": 0.0,
            "shi_acc": 0.0,
            "ge_acc": 0.0,
            "bai_top7_hit_rate": 0.0,
            "shi_top7_hit_rate": 0.0,
            "ge_top7_hit_rate": 0.0,
            "position_top7_all_hit_rate": 0.0,
            "full_hit_rate": 0.0,
            "top5_hit_rate": 0.0,
            "top10_hit_rate": 0.0,
            "top20_hit_rate": 0.0,
            "top50_hit_rate": 0.0,
        }

    counters = {
        "bai": 0,
        "shi": 0,
        "ge": 0,
        "bai_top7": 0,
        "shi_top7": 0,
        "ge_top7": 0,
        "position_top7_all": 0,
        "full": 0,
        "top5": 0,
        "top10": 0,
        "top20": 0,
        "top50": 0,
    }
    for actual, scores in zip(actual_numbers, score_rows):
        scores = normalize_scores(scores)
        order = np.argsort(scores)[::-1]
        top_code = CODES[order[0]]
        if str(top_code) == actual:
            counters["full"] += 1
        if actual in set(CODES[order[:5]]):
            counters["top5"] += 1
        if actual in set(CODES[order[:10]]):
            counters["top10"] += 1
        if actual in set(CODES[order[:20]]):
            counters["top20"] += 1
        if actual in set(CODES[order[:50]]):
            counters["top50"] += 1

        marginals = position_marginals_from_code_scores(scores)
        actual_digits = [int(actual[0]), int(actual[1]), int(actual[2])]
        if int(np.argmax(marginals[0])) == actual_digits[0]:
            counters["bai"] += 1
        if int(np.argmax(marginals[1])) == actual_digits[1]:
            counters["shi"] += 1
        if int(np.argmax(marginals[2])) == actual_digits[2]:
            counters["ge"] += 1
        top7_sets = [set(np.argsort(marginal)[::-1][:7].astype(int).tolist()) for marginal in marginals]
        top7_hits = [
            actual_digit in top7
            for actual_digit, top7 in zip(actual_digits, top7_sets)
        ]
        if top7_hits[0]:
            counters["bai_top7"] += 1
        if top7_hits[1]:
            counters["shi_top7"] += 1
        if top7_hits[2]:
            counters["ge_top7"] += 1
        if all(top7_hits):
            counters["position_top7_all"] += 1

    n = len(actual_numbers)
    return {
        "samples": n,
        "bai_acc": counters["bai"] / n,
        "shi_acc": counters["shi"] / n,
        "ge_acc": counters["ge"] / n,
        "bai_top7_hit_rate": counters["bai_top7"] / n,
        "shi_top7_hit_rate": counters["shi_top7"] / n,
        "ge_top7_hit_rate": counters["ge_top7"] / n,
        "position_top7_all_hit_rate": counters["position_top7_all"] / n,
        "full_hit_rate": counters["full"] / n,
        "top5_hit_rate": counters["top5"] / n,
        "top10_hit_rate": counters["top10"] / n,
        "top20_hit_rate": counters["top20"] / n,
        "top50_hit_rate": counters["top50"] / n,
    }


def ensemble_scores(model_scores: dict[str, np.ndarray], weights: dict[str, float] | None = None) -> np.ndarray:
    usable = {name: calibrate_model_scores(scores) for name, scores in model_scores.items() if name != "random_baseline"}
    if not usable:
        return np.ones(len(CODES), dtype=float) / len(CODES)

    if weights is None:
        weights = {name: 1.0 for name in usable}

    total_weight = sum(max(0.0, float(weights.get(name, 0.0))) for name in usable)
    if total_weight <= 0:
        weights = {name: 1.0 for name in usable}
        total_weight = float(len(usable))

    combined = np.zeros(len(CODES), dtype=float)
    for name, scores in usable.items():
        weight = max(0.0, float(weights.get(name, 0.0))) / total_weight
        combined += weight * scores
    return normalize_scores(combined)


def derive_weights_from_metrics(metrics: dict[str, dict[str, float]]) -> dict[str, float]:
    weights: dict[str, float] = {}
    for name, metric in metrics.items():
        if name == "random_baseline" or name.startswith("ensemble"):
            continue
        samples = float(metric.get("samples", 0.0))
        shrink = min(1.0, samples / 1000.0) if samples > 0 else 0.0
        top10_gain = float(metric.get("top10_hit_rate", 0.0)) - 0.01
        top7_all_gain = float(metric.get("position_top7_all_hit_rate", 0.0)) - 0.343
        single_top7 = (
            float(metric.get("bai_top7_hit_rate", 0.0))
            + float(metric.get("shi_top7_hit_rate", 0.0))
            + float(metric.get("ge_top7_hit_rate", 0.0))
        ) / 3.0
        single_top7_gain = single_top7 - 0.70
        signal = top10_gain * 6.0 + top7_all_gain * 1.5 + single_top7_gain * 0.8
        raw_weight = 1.0 + shrink * signal
        if samples < 1000:
            raw_weight = min(1.20, max(0.80, raw_weight))
        else:
            raw_weight = min(1.60, max(0.50, raw_weight))
        weights[name] = raw_weight

    total = sum(weights.values())
    if total <= 0:
        return {}
    return {name: value / total for name, value in weights.items()}


@dataclass
class RunConfig:
    seed: int = 42
    train_ratio: float = 0.8
    enable_boosting: bool = False
    torch_config: TorchConfig | None = None


def run_backtest(df: pd.DataFrame, feature_df: pd.DataFrame, config: RunConfig) -> tuple[dict[str, dict[str, float]], dict[str, float]]:
    print_section("时间顺序回测")
    feature_cols = get_feature_columns(feature_df)
    if len(feature_df) < 1000:
        print("当前样本量较少，模型结果仅适合学习和模拟研究。建议准备 1000 期以上数据后再观察长期稳定性。")

    split_at = max(5, int(len(feature_df) * config.train_ratio))
    split_at = min(split_at, len(feature_df) - 1)
    train_df = feature_df.iloc[:split_at].copy()
    test_df = feature_df.iloc[split_at:].copy()

    x_train = train_df[feature_cols].to_numpy(dtype=float)
    y_train = train_df[["target_bai", "target_shi", "target_ge"]].to_numpy(dtype=int)
    x_test = test_df[feature_cols].to_numpy(dtype=float)
    actual_numbers = test_df["target_number"].astype(str).tolist()

    print(f"训练样本：{len(train_df)}，测试样本：{len(test_df)}，切分方式：前 {config.train_ratio:.0%} 训练，后 {1 - config.train_ratio:.0%} 测试。")
    print("注意：未随机打乱数据，测试集在时间上晚于训练集。")

    model_score_rows: dict[str, list[np.ndarray]] = {}

    for row_number, original_idx in enumerate(test_df["row_index"].astype(int).tolist()):
        history = df.iloc[:original_idx]
        stat_scores = score_statistical_models(history, seed=config.seed + row_number)
        for name, scores in stat_scores.items():
            model_score_rows.setdefault(name, []).append(scores)

    ml_bundles = fit_ml_models(x_train, y_train, seed=config.seed, enable_boosting=config.enable_boosting)
    for name, bundle in ml_bundles.items():
        pos_probs = bundle.predict_position_probs(x_test)
        for row_idx in range(len(x_test)):
            row_probs = (pos_probs[0][row_idx], pos_probs[1][row_idx], pos_probs[2][row_idx])
            model_score_rows.setdefault(name, []).append(code_scores_from_position_probs(row_probs))

    torch_config = config.torch_config or TorchConfig(enabled=False)
    torch_probs = fit_torch_tabresnet(x_train, y_train, x_test, torch_config, seed=config.seed)
    if torch_probs is not None:
        for row_idx in range(len(x_test)):
            row_probs = (torch_probs[0][row_idx], torch_probs[1][row_idx], torch_probs[2][row_idx])
            model_score_rows.setdefault("torch_tabresnet", []).append(code_scores_from_position_probs(row_probs))

    metrics: dict[str, dict[str, float]] = {}
    for name, rows in model_score_rows.items():
        metrics[name] = evaluate_score_rows(actual_numbers, rows)

    ensemble_equal_rows: list[np.ndarray] = []
    for row_idx in range(len(test_df)):
        current_scores = {name: rows[row_idx] for name, rows in model_score_rows.items()}
        ensemble_equal_rows.append(ensemble_scores(current_scores))
    metrics["ensemble_equal"] = evaluate_score_rows(actual_numbers, ensemble_equal_rows)

    weights = derive_weights_from_metrics(metrics)

    print_backtest_table(metrics)
    if weights:
        print("\n按回测表现得到的模拟集成权重：")
        for name, weight in sorted(weights.items(), key=lambda item: item[1], reverse=True):
            print(f"- {name}: {weight:.3f}")
    else:
        print("\n没有得到有效的自动权重，后续候选排序将使用简单平均。")

    print("\n随机理论参考：百/十/个位单独命中约 10%，完整号码命中约 0.1%，Top10 约 1%，Top20 约 2%。")
    print(RANDOM_BASELINE_TEXT)
    return metrics, weights


def print_backtest_table(metrics: dict[str, dict[str, float]]) -> None:
    headers = ["模型", "样本", "百位", "十位", "个位", "完整", "Top5", "Top10", "Top20"]
    print("\n" + " | ".join(headers))
    print("-" * 86)
    for name, metric in sorted(metrics.items()):
        values = [
            name,
            str(int(metric["samples"])),
            f"{metric['bai_acc']:.3f}",
            f"{metric['shi_acc']:.3f}",
            f"{metric['ge_acc']:.3f}",
            f"{metric['full_hit_rate']:.3f}",
            f"{metric['top5_hit_rate']:.3f}",
            f"{metric['top10_hit_rate']:.3f}",
            f"{metric['top20_hit_rate']:.3f}",
        ]
        print(" | ".join(values))


def train_and_score_next(
    df: pd.DataFrame,
    feature_df: pd.DataFrame,
    config: RunConfig,
) -> dict[str, np.ndarray]:
    feature_cols = get_feature_columns(feature_df)
    x_train = feature_df[feature_cols].to_numpy(dtype=float)
    y_train = feature_df[["target_bai", "target_shi", "target_ge"]].to_numpy(dtype=int)
    next_feature = pd.DataFrame([build_history_feature(df)], columns=feature_cols).fillna(0.0)
    x_next = next_feature[feature_cols].to_numpy(dtype=float)

    model_scores = score_statistical_models(df, seed=config.seed)

    ml_bundles = fit_ml_models(x_train, y_train, seed=config.seed, enable_boosting=config.enable_boosting)
    for name, bundle in ml_bundles.items():
        probs = bundle.predict_position_probs(x_next)
        row_probs = (probs[0][0], probs[1][0], probs[2][0])
        model_scores[name] = code_scores_from_position_probs(row_probs)

    torch_config = config.torch_config or TorchConfig(enabled=False)
    torch_probs = fit_torch_tabresnet(x_train, y_train, x_next, torch_config, seed=config.seed)
    if torch_probs is not None:
        row_probs = (torch_probs[0][0], torch_probs[1][0], torch_probs[2][0])
        model_scores["torch_tabresnet"] = code_scores_from_position_probs(row_probs)

    return model_scores


def build_candidate_rows(
    combined_scores: np.ndarray,
    model_scores: dict[str, np.ndarray],
    top_k: int = 20,
) -> list[dict[str, object]]:
    order = np.argsort(combined_scores)[::-1][:top_k]
    rows: list[dict[str, object]] = []
    for rank, idx in enumerate(order, start=1):
        code = str(CODES[idx])
        desc = describe_code(code)
        support = {
            name: float(calibrate_model_scores(scores)[idx])
            for name, scores in model_scores.items()
            if name != "random_baseline"
        }
        rows.append(
            {
                "rank": rank,
                "number": code,
                "score": float(combined_scores[idx]),
                "sum": desc["sum"],
                "span": desc["span"],
                "type": desc["type"],
                "support": support,
            }
        )
    return rows


def print_candidate_table(rows: list[dict[str, object]], top_k: int = 20) -> None:
    print_section(f"Top {top_k} 模拟候选号码")
    print(SIMULATION_TEXT)
    print(SAFETY_TEXT)
    print("综合分数为模型相对分数，只用于排序，不是中奖概率。")
    print("\n排名 | 号码 | 综合分数 | 和值 | 跨度 | 类型 | 模型支持分数")
    print("-" * 100)
    for row in rows[:top_k]:
        support = row["support"]
        assert isinstance(support, dict)
        support_text = ", ".join(f"{name}:{value:.6f}" for name, value in sorted(support.items()))
        print(
            f"{row['rank']:>2} | {row['number']} | {row['score']:.6f} | "
            f"{row['sum']:>2} | {row['span']:>1} | {row['type']} | {support_text}"
        )


def print_data_summary(df: pd.DataFrame, csv_path: Path) -> None:
    print_section("数据概览")
    print(f"CSV 文件：{csv_path}")
    print(f"总期数：{len(df)}")
    print(f"日期范围：{df['date'].iloc[0]} 到 {df['date'].iloc[-1]}")
    print(f"最近 3 期：")
    for row in df.tail(3).itertuples(index=False):
        print(f"- {row.issue} | {row.date} | {row.number}")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="福彩3D模拟分析助手：学习、特征工程、回测和模型模拟候选输出。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--csv", default="fc3d_history.csv", help="历史开奖 CSV 路径，字段为 issue,date,number。")
    parser.add_argument("--create-sample", action="store_true", help="生成示例数据。")
    parser.add_argument("--sample-size", type=int, default=240, help="示例数据期数。")
    parser.add_argument("--overwrite", action="store_true", help="生成示例数据时允许覆盖已有 CSV。")
    parser.add_argument("--sample-only", action="store_true", help="只生成示例数据，不继续训练和回测。")
    parser.add_argument("--skip-backtest", action="store_true", help="跳过时间顺序回测。")
    parser.add_argument("--skip-predict", action="store_true", help="跳过模拟候选号码输出。")
    parser.add_argument("--top-k", type=int, default=20, help="输出 Top-K 模拟候选号码。")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="时间顺序切分中训练集比例。")
    parser.add_argument("--seed", type=int, default=42, help="随机种子。")
    parser.add_argument("--enable-boosting", action="store_true", help="如果已安装 XGBoost/LightGBM/CatBoost，则尝试启用。")
    parser.add_argument("--use-torch", action="store_true", help="如果已安装 PyTorch，则启用 TabResNet。")
    parser.add_argument("--epochs", type=int, default=12, help="TabResNet 训练轮数。")
    parser.add_argument("--batch-size", type=int, default=32, help="TabResNet batch size。")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="TabResNet 学习率。")
    parser.add_argument("--hidden-dim", type=int, default=128, help="TabResNet 隐藏层维度。")
    parser.add_argument("--dropout", type=float, default=0.15, help="TabResNet dropout。")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="TabResNet weight decay。")
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> None:
    if not 0.5 <= args.train_ratio < 1.0:
        raise ValueError("--train-ratio 建议在 [0.5, 1.0) 之间。")
    if args.top_k <= 0 or args.top_k > 1000:
        raise ValueError("--top-k 必须在 1 到 1000 之间。")
    if args.sample_size < 30:
        raise ValueError("--sample-size 至少建议为 30。")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    validate_args(args)
    print_safety()

    csv_path = Path(args.csv)
    if args.create_sample:
        generate_sample_data(csv_path, rows=args.sample_size, seed=args.seed, overwrite=args.overwrite)
        if args.sample_only:
            return 0
    elif not csv_path.exists():
        print(f"\n未找到 {csv_path}，为了让初学者能直接运行，程序会自动生成一份示例数据。")
        generate_sample_data(csv_path, rows=args.sample_size, seed=args.seed, overwrite=False)

    df = load_history_data(csv_path)
    print_data_summary(df, csv_path)
    feature_df = build_feature_table(df)

    if len(feature_df) < 20:
        print("可训练样本太少，复杂模型可能会自动退回到统计模型。")

    torch_config = TorchConfig(
        enabled=bool(args.use_torch),
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        weight_decay=args.weight_decay,
    )
    config = RunConfig(
        seed=args.seed,
        train_ratio=args.train_ratio,
        enable_boosting=bool(args.enable_boosting),
        torch_config=torch_config,
    )

    backtest_weights: dict[str, float] = {}
    if not args.skip_backtest:
        _, backtest_weights = run_backtest(df, feature_df, config)

    if not args.skip_predict:
        print_section("训练全量数据并输出模拟候选")
        model_scores = train_and_score_next(df, feature_df, config)
        if backtest_weights:
            combined = ensemble_scores(model_scores, weights=backtest_weights)
            print("已使用回测表现自动权重进行加权平均。")
        else:
            combined = ensemble_scores(model_scores)
            print("已使用简单平均进行模型集成。")

        top_k = min(args.top_k, 1000)
        rows = build_candidate_rows(combined, model_scores, top_k=top_k)
        print_candidate_table(rows, top_k=top_k)
        if top_k >= 10:
            print("\nTop 10 模拟候选号码：", ", ".join(str(row["number"]) for row in rows[:10]))
        if top_k >= 20:
            print("Top 20 模拟候选号码：", ", ".join(str(row["number"]) for row in rows[:20]))

    print("\n" + SAFETY_TEXT)
    print(SIMULATION_TEXT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

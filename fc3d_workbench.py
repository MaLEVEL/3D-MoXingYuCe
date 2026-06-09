#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
福彩3D模拟分析工作台

启动方式:
    streamlit run fc3d_workbench.py

说明:
    - 支持上传 CSV
    - 支持数据校验与清洗
    - 支持统计基线、传统机器学习、可选 TabResNet
    - 支持时间顺序回测与候选号码排序
    - 所有输出仅用于学习和模拟研究，不代表真实开奖结果
"""

from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import math
import re
import time
import urllib.parse
import urllib.request
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timedelta
from html import escape, unescape
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

try:
    import sklearn  # noqa: F401

    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False

APP_TITLE = "福彩3D模拟分析工作台"
APP_SUBTITLE = "导入历史数据，比较多个模型，做严格时间回测，仅供学习。"
BUILTIN_HISTORY_PATH = Path(__file__).with_name("fc3d_last_5_years_history.csv")
HUINIAO_API_URL = "https://api.huiniao.top/interface/home/lotteryHistory"
HUINIAO_API_FROM = "uf6veq0x7mh9a9n"
IP138_3D_URL = "https://caipiao.ip138.com/3d/"
THREED178_YEAR_URL = "https://www.3d178.cn/kaijiang/{year}/"
APIHZ_LATEST_URL = "https://cn.apihz.cn/api/caipiao/fucai3d.php?id=88888888&key=88888888"
FALLBACK_17500_ASC_URL = "https://data.17500.cn/3d_asc.txt"
HISTORY_LOOKBACK_DAYS = 365 * 5
RECENT_HISTORY_LOOKBACK_DAYS = 14
DEFAULT_ROLLING_INITIAL_TRAIN_SIZE = 1000
DEFAULT_ROLLING_BATCH_SIZE = 50
MIN_STABLE_HISTORY_ROWS = 1000
MIN_DEEP_MODEL_TRAIN_SAMPLES = 1000
SAFETY_TEXT = "本工具仅用于学习和模拟研究，不代表真实预测，不建议用于投注。"
SIMULATION_TEXT = "仅为模型模拟，不代表真实开奖结果。"
RANDOM_BASELINE_TEXT = "理论随机基线：单个位 Top7 约 70.0%，三位同时落入7码集合约 34.3%；直选 Top10 约 1.00%，Top20 约 2.00%，Top50 约 5.00%。小样本回测只能作为演示参考。"
TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
THEME_OPTIONS = ("亮色", "暗色")
THEME_TOKENS = {
    "亮色": {
        "--bg": "#F4F7FB",
        "--sidebar-bg": "#FFFFFF",
        "--panel": "#FFFFFF",
        "--panel-2": "#F8FAFC",
        "--border": "#CBD5E1",
        "--text": "#0B1220",
        "--muted": "#334155",
        "--accent": "#1D4ED8",
        "--accent-2": "#0F766E",
        "--danger": "#B91C1C",
        "--ok": "#166534",
        "--warning": "#92400E",
        "--warning-bg": "#FFF3C4",
        "--warning-border": "#D97706",
        "--bad-bg": "#FEE2E2",
        "--bad-text": "#7F1D1D",
        "--good-bg": "#DCFCE7",
        "--good-text": "#14532D",
        "--secondary-bg": "#DBEAFE",
        "--secondary-text": "#1E3A8A",
        "--button-bg": "#E2E8F0",
        "--button-text": "#0F172A",
        "--primary-button-text": "#FFFFFF",
        "--validate-bg": "#DBEAFE",
        "--validate-text": "#1E3A8A",
        "--candidate-bg": "#0F766E",
        "--candidate-text": "#FFFFFF",
        "--solid-button-bg": "#172033",
        "--solid-button-text": "#FFFFFF",
        "--disabled-bg": "#E5E7EB",
        "--disabled-text": "#475569",
        "--shadow": "rgba(15, 23, 42, 0.10)",
        "--code-bg": "#111827",
        "--code-text": "#F8FAFC",
    },
    "暗色": {
        "--bg": "#090E1A",
        "--sidebar-bg": "#0F172A",
        "--panel": "#151C2E",
        "--panel-2": "#1E293B",
        "--border": "#475569",
        "--text": "#F8FAFC",
        "--muted": "#CBD5E1",
        "--accent": "#60A5FA",
        "--accent-2": "#2DD4BF",
        "--danger": "#FCA5A5",
        "--ok": "#86EFAC",
        "--warning": "#FDE68A",
        "--warning-bg": "#3A2A0A",
        "--warning-border": "#F59E0B",
        "--bad-bg": "#3B1212",
        "--bad-text": "#FECACA",
        "--good-bg": "#0F2E1D",
        "--good-text": "#BBF7D0",
        "--secondary-bg": "#1E3A5F",
        "--secondary-text": "#DBEAFE",
        "--button-bg": "#243247",
        "--button-text": "#F8FAFC",
        "--primary-button-text": "#06111F",
        "--validate-bg": "#1E3A5F",
        "--validate-text": "#DBEAFE",
        "--candidate-bg": "#2DD4BF",
        "--candidate-text": "#FFFFFF",
        "--solid-button-bg": "#334155",
        "--solid-button-text": "#F8FAFC",
        "--disabled-bg": "#1F2937",
        "--disabled-text": "#CBD5E1",
        "--shadow": "rgba(0, 0, 0, 0.35)",
        "--code-bg": "#020617",
        "--code-text": "#E2E8F0",
    },
}
THEORETICAL_TOP10_BASELINE = 10 / 1000
THEORETICAL_TOP20_BASELINE = 20 / 1000
THEORETICAL_TOP50_BASELINE = 50 / 1000
THEORETICAL_POSITION_TOP7_BASELINE = 7 / 10
THEORETICAL_POSITION_TOP7_ALL_BASELINE = 343 / 1000
THEORETICAL_DANMA_AT_LEAST_1_BASELINE = 1 - (7 / 10) ** 3
THEORETICAL_DANMA_AT_LEAST_2_BASELINE = 3 * (3 / 10) ** 2 * (7 / 10) + (3 / 10) ** 3
THEORETICAL_DANMA_ALL_3_POSITIONS_BASELINE = (3 / 10) ** 3
THEORETICAL_DANMA_ALL_3_UNIQUE_BASELINE = 6 / 1000
THEORETICAL_DANMA_AVERAGE_POSITION_HITS_BASELINE = 3 * (3 / 10)
THEORETICAL_NO_POSITION_7_AT_LEAST_1_BASELINE = 1 - (3 / 10) ** 3
THEORETICAL_NO_POSITION_7_AT_LEAST_2_BASELINE = 3 * (7 / 10) ** 2 * (3 / 10) + (7 / 10) ** 3
THEORETICAL_NO_POSITION_7_ALL_3_BASELINE = (7 / 10) ** 3
THEORETICAL_NO_POSITION_7_AVERAGE_POSITION_HITS_BASELINE = 3 * (7 / 10)
NO_POSITION_7_TARGET_23_RATE = 0.82
DANMA_INEFFECTIVE_TEXT = "当前胆码模型暂未证明长期有效，仅可作为模拟排序参考。"
DANMA_SCORE_NOTE = "模型相对分数不是中奖概率，只用于排序。"
DANMA_COMPONENT_COLUMNS = [
    "position_marginal_score",
    "unique_presence_score",
    "combination_marginal_score",
    "near30_frequency",
    "near100_frequency",
    "position_frequency",
    "omission_rebound",
    "sum_distribution_support",
    "heat_cold_stability",
]
DANMA_SCORE_WEIGHTS = {
    "combination_marginal_score": 0.1,
    "near30_frequency": 0.3,
    "near100_frequency": 0.0,
    "position_frequency": 0.3,
    "omission_rebound": 0.3,
    "sum_distribution_support": 0.0,
    "heat_cold_stability": 0.0,
}
NO_POSITION_7_SCORE_WEIGHTS = {
    "combination_marginal_score": 0.0435,
    "near30_frequency": 0.0,
    "near100_frequency": 0.0435,
    "position_frequency": 0.0435,
    "omission_rebound": 0.2174,
    "sum_distribution_support": 0.3478,
    "heat_cold_stability": 0.3043,
}
DIRECT_TOP20_FULL_COVERAGE_WEIGHTS = {
    "history_frequency": 0.12,
    "bayes_smooth_frequency": 0.26,
    "omission": 0.12,
    "sum_distribution": 0.38,
    "position_frequency": 0.12,
}
DIRECT_TOP20_FULL_COVERAGE_RULE = {
    "weights": dict(DIRECT_TOP20_FULL_COVERAGE_WEIGHTS),
    "validation_rate": 14 / 351,
    "validation_hits": 14,
    "validation_selected": 351,
    "validation_samples": 351,
}
NO_POSITION_7_CONFIDENCE_RULE = {
    "model": "omission",
    "feature": "spread",
    "direction": "<=",
    "threshold": 0.462038,
    "weights": dict(NO_POSITION_7_SCORE_WEIGHTS),
    "target_rate": 0.82,
    "validation_rate": 69 / 75,
    "validation_hits": 69,
    "validation_selected": 75,
    "validation_samples": 351,
    "tune_rate": 78 / 86,
    "tune_hits": 78,
    "tune_selected": 86,
    "tune_samples": 405,
}
POSITION_7_TARGET_ALL_RATE = 0.40
POSITION_7_SCORE_WEIGHTS = {
    "history_frequency": 0.0,
    "bayes_smooth_frequency": 0.1,
    "omission": 0.7,
    "sum_distribution": 0.0,
    "position_frequency": 0.2,
}
POSITION_7_CONFIDENCE_RULE = {
    "model": "position_weighted",
    "feature": "full_coverage",
    "direction": "all",
    "threshold": None,
    "weights": dict(POSITION_7_SCORE_WEIGHTS),
    "target_rate": POSITION_7_TARGET_ALL_RATE,
    "validation_rate": 139 / 351,
    "validation_hits": 139,
    "validation_selected": 351,
    "validation_samples": 351,
    "validation_metric": "三位全中",
    "at_least_2_rate": 288 / 351,
    "at_least_2_hits": 288,
    "tune_rate": 142 / 405,
    "tune_hits": 142,
    "tune_selected": 405,
    "tune_samples": 405,
}
DANMA_CONFIDENCE_RULE = {
    "model": "omission",
    "feature": "bottom_sum",
    "direction": ">=",
    "threshold": 3.071032,
    "weights": dict(DANMA_SCORE_WEIGHTS),
    "target_rate": THEORETICAL_DANMA_AT_LEAST_1_BASELINE + 0.02,
    "validation_rate": 86 / 109,
    "validation_hits": 86,
    "validation_selected": 109,
    "validation_samples": 351,
    "tune_rate": 67 / 86,
    "tune_hits": 67,
    "tune_selected": 86,
    "tune_samples": 405,
}
DANMA_COMPONENT_LABELS = {
    "combination_marginal_score": "组合边际",
    "near30_frequency": "近30期频率",
    "near100_frequency": "近100期频率",
    "position_frequency": "位置频率",
    "omission_rebound": "遗漏值回补",
    "sum_distribution_support": "和值分布",
    "heat_cold_stability": "冷热稳定性",
}
DANMA_BACKTEST_SPECS = [
    ("至少命中1个位置", "danma_at_least_1_position_hit_rate", THEORETICAL_DANMA_AT_LEAST_1_BASELINE),
    ("至少命中2个位置", "danma_at_least_2_position_hit_rate", THEORETICAL_DANMA_AT_LEAST_2_BASELINE),
    ("三位都落入胆码集合", "danma_all_3_positions_in_set_rate", THEORETICAL_DANMA_ALL_3_POSITIONS_BASELINE),
    ("3个胆码全部出现在开奖号码去重集合", "danma_all_3_danma_in_unique_draw_rate", THEORETICAL_DANMA_ALL_3_UNIQUE_BASELINE),
]
NO_POSITION_7_BACKTEST_SPECS = [
    ("7码>=1", "no_position_7_at_least_1_position_hit_rate", THEORETICAL_NO_POSITION_7_AT_LEAST_1_BASELINE),
    ("7码=23", "no_position_7_at_least_2_position_hit_rate", THEORETICAL_NO_POSITION_7_AT_LEAST_2_BASELINE),
    ("7码三位全包", "no_position_7_all_3_positions_in_set_rate", THEORETICAL_NO_POSITION_7_ALL_3_BASELINE),
]

BASELINE_MODELS = [
    "random_baseline",
    "history_frequency",
    "bayes_smooth_frequency",
    "omission",
    "sum_distribution",
    "position_frequency",
]

ML_MODEL_LABELS = {
    "random_baseline": "理论随机基线",
    "history_frequency": "历史频率",
    "bayes_smooth_frequency": "贝叶斯平滑",
    "omission": "遗漏值",
    "sum_distribution": "和值分布",
    "position_frequency": "位置频率",
    "logistic_regression": "逻辑回归",
    "random_forest": "随机森林",
    "xgboost": "极端梯度提升",
    "lightgbm": "轻量梯度提升",
    "catboost": "类别提升",
    "tabresnet": "残差表格网络",
    "ensemble_equal": "等权集成",
    "position_weighted": "定位7码加权",
}

DEFAULT_WINDOWS = [30, 60, 120]
_CORE_MODULE: Any | None = None


def get_core() -> Any:
    global _CORE_MODULE
    if _CORE_MODULE is None:
        import fc3d_ai_assistant as core_module

        _CORE_MODULE = core_module
    return _CORE_MODULE


@dataclass
class AppState:
    file_name: str = ""
    data_version: str = ""
    last_loaded_at: str = ""
    last_trained_at: str = ""
    last_backtest_at: str = ""
    last_predict_at: str = ""
    current_mode: str = "未训练"


def now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def short_hash(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()[:10]


def full_hash(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def normalize_theme_mode(mode: str | None) -> str:
    return str(mode) if mode in THEME_OPTIONS else THEME_OPTIONS[0]


def theme_css_variables(mode: str | None) -> str:
    tokens = THEME_TOKENS[normalize_theme_mode(mode)]
    return "\n".join(f"          {name}: {value};" for name, value in tokens.items())


def active_theme_mode() -> str:
    return normalize_theme_mode(st.session_state.get("app_theme_mode", THEME_OPTIONS[0]))


def init_state() -> None:
    defaults: dict[str, Any] = {
        "logs": [],
        "raw_bytes": None,
        "raw_df": None,
        "clean_df": None,
        "validation_report": None,
        "feature_df": None,
        "trained_models": None,
        "backtest_metrics": None,
        "backtest_weights": None,
        "backtest_note": "",
        "backtest_fold_df": None,
        "candidate_df": None,
        "candidate_groups": None,
        "candidate_pool_df": None,
        "candidate_model_scores": None,
        "position_7_report": None,
        "danma_prediction": None,
        "no_position_7_prediction": None,
        "active_no_position_7_rule": deepcopy(NO_POSITION_7_CONFIDENCE_RULE),
        "active_position_7_rule": deepcopy(POSITION_7_CONFIDENCE_RULE),
        "active_danma_rule": deepcopy(DANMA_CONFIDENCE_RULE),
        "app_theme_mode": THEME_OPTIONS[0],
        "app_state": AppState(),
        "active_tab": "数据",
        "last_report_text": "",
        "last_report_file": "fc3d_report.md",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def log_event(message: str) -> None:
    if "logs" not in st.session_state:
        st.session_state.logs = []
    stamp = now_str()
    st.session_state.logs.append(f"[{stamp}] {message}")
    st.session_state.logs = st.session_state.logs[-200:]


def get_active_no_position_7_rule() -> dict[str, Any]:
    return deepcopy(st.session_state.get("active_no_position_7_rule", NO_POSITION_7_CONFIDENCE_RULE))


def get_active_position_7_rule() -> dict[str, Any]:
    return deepcopy(st.session_state.get("active_position_7_rule", POSITION_7_CONFIDENCE_RULE))


def get_active_danma_rule() -> dict[str, Any]:
    return deepcopy(st.session_state.get("active_danma_rule", DANMA_CONFIDENCE_RULE))


def apply_quick_prediction_scheme(scheme_key: str) -> str:
    if scheme_key == "no_position_7_coverage":
        st.session_state.active_no_position_7_rule = deepcopy(NO_POSITION_7_CONFIDENCE_RULE)
        st.session_state.no_position_7_prediction = None
        log_event("已应用不定位7码覆盖优先阈值方案。")
        return "已应用不定位7码覆盖优先阈值方案。"
    if scheme_key == "position_7_high_confidence":
        st.session_state.active_position_7_rule = deepcopy(POSITION_7_CONFIDENCE_RULE)
        st.session_state.position_7_report = None
        st.session_state.candidate_pool_df = None
        log_event("已应用定位7码高置信全中方案。")
        return "已应用定位7码高置信全中方案。"
    if scheme_key == "danma_optimal":
        st.session_state.active_danma_rule = deepcopy(DANMA_CONFIDENCE_RULE)
        st.session_state.danma_prediction = None
        log_event("已应用三胆码最优独立方案。")
        return "已应用三胆码最优独立方案。"
    if scheme_key == "direct_top20_full_coverage":
        st.session_state.backtest_weights = deepcopy(DIRECT_TOP20_FULL_COVERAGE_WEIGHTS)
        st.session_state.candidate_df = None
        st.session_state.candidate_groups = None
        st.session_state.candidate_pool_df = None
        st.session_state.position_7_report = None
        st.session_state.danma_prediction = None
        st.session_state.no_position_7_prediction = None
        log_event("已应用直选20注全覆盖权重方案。")
        return "已应用直选20注全覆盖权重方案。"
    raise ValueError(f"未知快捷方案：{scheme_key}")


def display_model_name(name: str) -> str:
    return ML_MODEL_LABELS.get(name, name)


def display_model_list(names: list[str] | tuple[str, ...] | set[str]) -> str:
    if not names:
        return "无"
    return "、".join(display_model_name(name) for name in sorted(names))


def selected_enabled_model_names(fit_deep: bool) -> list[str]:
    names = [name for name, enabled in selected_model_flags().items() if enabled]
    if not fit_deep and "tabresnet" in names:
        names.remove("tabresnet")
    return names


def should_fit_deep_model(train_samples: int, fit_deep: bool) -> bool:
    return bool(fit_deep and TORCH_AVAILABLE and train_samples >= MIN_DEEP_MODEL_TRAIN_SAMPLES)


def skipped_model_reasons(fit_deep: bool, train_samples: int | None = None) -> list[str]:
    skipped: list[str] = []
    optional_packages = {
        "xgboost": "xgboost",
        "lightgbm": "lightgbm",
        "catboost": "catboost",
    }
    for model_name, package_name in optional_packages.items():
        if st.session_state.get(f"use_{model_name}", False) and importlib.util.find_spec(package_name) is None:
            skipped.append(f"{display_model_name(model_name)}：环境未安装 {package_name}")
    if fit_deep and not TORCH_AVAILABLE:
        skipped.append("残差表格网络：环境未安装 PyTorch")
    if fit_deep and TORCH_AVAILABLE and train_samples is not None and train_samples < MIN_DEEP_MODEL_TRAIN_SAMPLES:
        skipped.append(f"残差表格网络：训练样本少于 {MIN_DEEP_MODEL_TRAIN_SAMPLES}，已跳过以降低过拟合风险")
    return skipped


def log_run_details(action: str, elapsed: float, model_scores: dict[str, np.ndarray] | None = None, metrics: dict[str, dict[str, float]] | None = None, skipped: list[str] | None = None) -> None:
    parts = [f"{action}耗时 {elapsed:.2f} 秒"]
    if model_scores:
        visible_models = [name for name in active_score_items(model_scores)]
        if action == "候选生成":
            parts.append(f"候选生成使用模型：{display_model_list(visible_models)}")
            parts.append("对比基线：三位7码34.3%，Top10为1.0%，Top20为2.0%")
        else:
            parts.append(f"使用模型：{display_model_list(visible_models)}")
    if metrics:
        visible_metrics = [name for name in metrics if name != "random_baseline"]
        parts.append(f"使用模型：{display_model_list(visible_metrics)}")
        sample_count = max((int(metric.get("samples", 0)) for name, metric in metrics.items() if name != "random_baseline"), default=0)
        parts.append(f"回测样本数量 {sample_count}")
    if skipped:
        parts.append(f"跳过模型：{'；'.join(skipped)}")
    log_event("；".join(parts) + "。")


def app_stage() -> str:
    if st.session_state.candidate_df is not None:
        return "候选已生成"
    if st.session_state.backtest_metrics:
        return "回测完成"
    if st.session_state.trained_models is not None:
        return "训练完成"
    if st.session_state.validation_report:
        return "已校验"
    if st.session_state.raw_bytes is not None:
        return "已加载未校验"
    return "未加载数据"


def sync_current_mode() -> None:
    st.session_state.app_state.current_mode = app_stage()


def action_availability() -> dict[str, bool]:
    stage = app_stage()
    return {
        "demo": True,
        "validate": stage == "已加载未校验",
        "train": stage in {"已校验", "训练完成", "回测完成", "候选已生成"},
        "backtest": stage in {"训练完成", "回测完成", "候选已生成"},
        "predict": stage in {"训练完成", "回测完成", "候选已生成"},
        "export": stage in {"回测完成", "候选已生成"},
    }


def recommended_action_key() -> str:
    stage = app_stage()
    if stage == "未加载数据":
        return "demo"
    if stage == "已加载未校验":
        return "validate"
    if stage == "已校验":
        return "train"
    if stage == "训练完成":
        return "backtest"
    if stage == "回测完成":
        return "predict"
    if stage == "候选已生成":
        return "export"
    return "demo"


def next_step_copy() -> tuple[str, str]:
    stage = app_stage()
    copies = {
        "未加载数据": ("加载近5年开奖记录", "也可以在侧边栏上传自己的历史开奖CSV。"),
        "已加载未校验": ("校验数据", "检查字段、日期、重复期号和号码格式。"),
        "已校验": ("训练模型", "使用已校验数据训练当前启用的模型。"),
        "训练完成": ("开始滚动回测", "先看时间顺序回测，再判断排序是否有参考价值。"),
        "回测完成": ("生成候选与胆码", "生成下一期3个胆码、三个位7码和候选分组。"),
        "候选已生成": ("导出分析报告", "可查看候选、胆码，也可以导出报告和CSV。"),
    }
    return copies.get(stage, copies["未加载数据"])


def disabled_action_reason(action_key: str) -> str:
    stage = app_stage()
    if action_key == "validate":
        return "请先加载近5年开奖记录或上传CSV。"
    if action_key == "train":
        return "请先加载并校验数据。"
    if action_key == "backtest":
        return "请先训练模型。"
    if action_key == "predict":
        return "请先完成训练模型或滚动回测。"
    if action_key == "export":
        return "请先完成滚动回测并生成候选与胆码。"
    if action_key == "demo" and stage != "未加载数据":
        return "重新加载近5年开奖记录会清空当前训练、回测和候选结果。"
    return ""


def action_button_type(action_key: str, availability: dict[str, bool]) -> str:
    if action_key == recommended_action_key() and availability.get(action_key, False):
        return "primary"
    return "secondary"


def demo_reload_required() -> bool:
    return any(
        st.session_state.get(key) is not None
        for key in (
            "raw_bytes",
            "validation_report",
            "trained_models",
            "backtest_metrics",
            "candidate_df",
            "candidate_pool_df",
            "position_7_report",
        )
    )


def default_history_name() -> str:
    return "近5年开奖记录"


def fetch_url_text(url: str, timeout: int = 20) -> str:
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Referer": urllib.parse.urljoin(url, "/"),
        },
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        raw = response.read()
        declared_encoding = response.headers.get_content_charset()
    for encoding in [declared_encoding, "utf-8", "gb18030", "gbk"]:
        if not encoding:
            continue
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def normalize_history_rows(
    rows: list[dict[str, str]],
    days: int,
    source_name: str,
    min_rows: int = MIN_STABLE_HISTORY_ROWS,
) -> pd.DataFrame:
    if not rows:
        raise ValueError(f"{source_name}没有返回可用历史数据。")

    df = pd.DataFrame(rows)
    df["issue"] = df["issue"].astype(str).str.strip()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["number"] = df["number"].astype(str).str.strip().str.replace(r"\D", "", regex=True)
    df = df.dropna(subset=["date"])
    df = df[df["issue"].str.fullmatch(r"\d{6,8}")]
    df = df[df["number"].str.fullmatch(r"\d{3}")]
    df = df.drop_duplicates("issue", keep="last")
    if df.empty:
        raise ValueError(f"{source_name}没有可解析的期号、日期、开奖号。")

    latest_day = df["date"].max()
    start_day = latest_day - timedelta(days=days)
    df = df[(df["date"] >= start_day) & (df["date"] <= latest_day)]
    df = df.sort_values(["date", "issue"]).reset_index(drop=True)
    if len(df) < min_rows:
        raise ValueError(f"{source_name}数据不足：仅 {len(df)} 期。")

    df["date"] = df["date"].dt.strftime("%Y-%m-%d")
    return df[["issue", "date", "number"]]


def fetch_huiniao_history(
    days: int = HISTORY_LOOKBACK_DAYS,
    limit: int = 2500,
    min_rows: int = MIN_STABLE_HISTORY_ROWS,
) -> pd.DataFrame:
    params = {
        "type": "fcsd",
        "page": 1,
        "limit": limit,
        "from": HUINIAO_API_FROM,
    }
    url = f"{HUINIAO_API_URL}?{urllib.parse.urlencode(params)}"
    with urllib.request.urlopen(url, timeout=20) as response:
        payload = json.loads(response.read().decode("utf-8"))
    if int(payload.get("code", 0)) != 1:
        raise ValueError(f"历史开奖接口返回异常：{payload.get('info', '未知错误')}")

    data = payload.get("data", {})
    latest = data.get("last") or {}
    latest_day = pd.to_datetime(latest.get("day"), errors="coerce")
    if pd.isna(latest_day):
        raise ValueError("历史开奖接口缺少最新开奖日期。")
    start_day = latest_day.date() - timedelta(days=days)

    rows: list[dict[str, str]] = []
    for item in (data.get("data", {}).get("list") or []):
        day = pd.to_datetime(item.get("day"), errors="coerce")
        if pd.isna(day) or day.date() < start_day or day.date() > latest_day.date():
            continue
        try:
            digits = [int(item["one"]), int(item["two"]), int(item["three"])]
        except Exception as exc:
            raise ValueError(f"历史开奖接口号码字段异常：{item}") from exc
        if any(digit < 0 or digit > 9 for digit in digits):
            raise ValueError(f"历史开奖接口号码不在 0-9 范围：{item}")
        rows.append(
            {
                "issue": str(item.get("code", "")).strip(),
                "date": day.strftime("%Y-%m-%d"),
                "number": "".join(str(digit) for digit in digits),
            }
        )

    return normalize_history_rows(rows, days, "Huiniao API", min_rows=min_rows)


def fetch_huiniao_recent_history(days: int = RECENT_HISTORY_LOOKBACK_DAYS, limit: int = 60) -> pd.DataFrame:
    return fetch_huiniao_history(days=days, limit=limit, min_rows=1)


def fetch_3d178_history(days: int = HISTORY_LOOKBACK_DAYS) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    current_year = datetime.now().year
    for year in range(current_year - 5, current_year + 1):
        base_url = THREED178_YEAR_URL.format(year=year)
        first_page = unescape(fetch_url_text(base_url))
        page_urls = {base_url}
        for link in re.findall(r'href=["\'](index\d*\.shtml)["\']', first_page):
            page_urls.add(urllib.parse.urljoin(base_url, link))

        for page_url in sorted(page_urls):
            if page_url == base_url:
                text = first_page
            else:
                try:
                    text = unescape(fetch_url_text(page_url))
                except Exception:
                    continue
            for match in re.finditer(
                r'<td\s+class="td_qh">.*?>(\d{6,8})</a>.*?'
                r'<td\s+class="td_code">\s*<span>(\d)</span><span>(\d)</span><span>(\d)</span>.*?'
                r"<td>(\d{4}-\d{2}-\d{2})</td>",
                text,
                re.S,
            ):
                issue, one, two, three, day = match.groups()
                rows.append({"issue": issue, "date": day, "number": f"{one}{two}{three}"})
            for match in re.finditer(
                r'<td\s+class="kjdate"[^>]*>\s*(\d{4}-\d{2}-\d{2})\s*</td>\s*'
                r"<td>\s*(\d{6,8})\s*</td>\s*"
                r'<td\s+class="red">\s*(\d)\s+(\d)\s+(\d)\s*</td>',
                text,
                re.S,
            ):
                day, issue, one, two, three = match.groups()
                rows.append({"issue": issue, "date": day, "number": f"{one}{two}{three}"})

    return normalize_history_rows(rows, days, "3D之家年度页")


def fetch_ip138_history(days: int = HISTORY_LOOKBACK_DAYS) -> pd.DataFrame:
    text = unescape(fetch_url_text(IP138_3D_URL))
    rows: list[dict[str, str]] = []
    for match in re.finditer(
        r"<tr>.*?<td>\s*<span>(\d{4}-\d{2}-\d{2})</span>\s*</td>\s*"
        r"<td><span>(\d{6,8})</span></td>\s*"
        r'<td\s+class="award">\s*'
        r'<span[^>]*data-value="(\d)"[^>]*>.*?</span>\s*'
        r'<span[^>]*data-value="(\d)"[^>]*>.*?</span>\s*'
        r'<span[^>]*data-value="(\d)"[^>]*>.*?</span>',
        text,
        re.S,
    ):
        day, issue, one, two, three = match.groups()
        rows.append({"issue": issue, "date": day, "number": f"{one}{two}{three}"})

    return normalize_history_rows(rows, days, "IP138开奖页")


def fetch_17500_history(days: int = HISTORY_LOOKBACK_DAYS) -> pd.DataFrame:
    text = fetch_url_text(FALLBACK_17500_ASC_URL)
    rows: list[dict[str, str]] = []
    for line in text.splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        if not re.fullmatch(r"\d{6,8}", parts[0]):
            continue
        day = pd.to_datetime(parts[1], errors="coerce")
        if pd.isna(day):
            continue
        digits = parts[2:5]
        if any(not re.fullmatch(r"\d", digit) for digit in digits):
            continue
        rows.append(
            {
                "issue": parts[0],
                "date": day.strftime("%Y-%m-%d"),
                "number": "".join(digits),
            }
        )

    return normalize_history_rows(rows, days, "17500历史文本源")


def fetch_apihz_latest_draw() -> dict[str, str]:
    payload = json.loads(fetch_url_text(APIHZ_LATEST_URL))
    if int(payload.get("code", 0)) != 200:
        raise ValueError(payload.get("msg", "APIHz最新开奖接口返回异常"))
    digits = re.findall(r"\d", str(payload.get("number", "")))
    day = str(payload.get("time", "")).split("(")[0].strip()
    if len(digits) != 3 or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", day):
        raise ValueError("APIHz最新开奖接口缺少可解析的日期或开奖号。")
    return {"issue": str(payload.get("qihao", "")).strip(), "date": day, "number": "".join(digits)}


def fetch_online_history(days: int = HISTORY_LOOKBACK_DAYS) -> tuple[pd.DataFrame, str]:
    errors: list[str] = []
    sources: list[tuple[str, Callable[[], pd.DataFrame]]] = [
        ("17500历史文本源", lambda: fetch_17500_history(days=days)),
        ("3D之家年度页", lambda: fetch_3d178_history(days=days)),
        ("Huiniao API", lambda: fetch_huiniao_history(days=days)),
        ("IP138开奖页", lambda: fetch_ip138_history(days=days)),
    ]
    for source_name, fetcher in sources:
        try:
            return fetcher(), source_name
        except Exception as exc:
            errors.append(f"{source_name}：{exc}")

    try:
        latest = fetch_apihz_latest_draw()
        errors.append(f"APIHz最新开奖接口：仅返回最新一期 {latest['issue']}，不能补齐近5年历史。")
    except Exception as exc:
        errors.append(f"APIHz最新开奖接口：{exc}")
    raise ValueError("所有在线数据源获取失败；" + "；".join(errors))


def fetch_recent_online_history(days: int = RECENT_HISTORY_LOOKBACK_DAYS) -> tuple[pd.DataFrame, str]:
    recent_days = max(1, int(days))
    errors: list[str] = []
    sources: list[tuple[str, Callable[[], pd.DataFrame]]] = [
        ("Huiniao API近期开奖", lambda: fetch_huiniao_recent_history(days=recent_days, limit=max(30, recent_days * 4))),
        ("APIHz最新开奖接口", lambda: pd.DataFrame([fetch_apihz_latest_draw()])),
    ]
    for source_name, fetcher in sources:
        try:
            return fetcher(), source_name
        except Exception as exc:
            errors.append(f"{source_name}：{exc}")

    raise ValueError("所有近期数据源获取失败；" + "；".join(errors))


def normalize_repository_dataframe(df: pd.DataFrame, source_name: str) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["issue", "date", "number"])

    required_cols = ["issue", "date", "number"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"{source_name}缺少字段：{missing_cols}")

    out = df[required_cols].copy()
    out["issue"] = out["issue"].astype("string").str.strip()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out["number"] = out["number"].astype("string").str.strip().str.replace(r"\D", "", regex=True).str.zfill(3)
    out = out.dropna(subset=["date"])
    out = out[out["issue"].str.fullmatch(r"\d{5,8}").fillna(False)]
    out = out[out["number"].str.fullmatch(r"\d{3}").fillna(False)]
    out = out.drop_duplicates("issue", keep="last")
    out = out.sort_values(["date", "issue"]).reset_index(drop=True)
    out["date"] = out["date"].dt.strftime("%Y-%m-%d")
    return out[required_cols]


def update_history_repository(
    history_path: Path | str = BUILTIN_HISTORY_PATH,
    days: int = HISTORY_LOOKBACK_DAYS,
) -> dict[str, Any]:
    path = Path(history_path)
    if path.exists() and path.stat().st_size > 0:
        local_clean = normalize_repository_dataframe(read_csv_bytes(path.read_bytes()), path.name)
    else:
        local_clean = pd.DataFrame(columns=["issue", "date", "number"])

    if local_clean.empty:
        online_df, source_name = fetch_online_history(days=days)
        incremental_mode = False
    else:
        local_dates = pd.to_datetime(local_clean["date"], errors="coerce")
        latest_local_day = local_dates.max()
        if pd.isna(latest_local_day):
            recent_days = RECENT_HISTORY_LOOKBACK_DAYS
        else:
            today = pd.Timestamp.now().normalize()
            missing_days = max(1, int((today - latest_local_day.normalize()).days) + 2)
            recent_days = min(int(days), max(RECENT_HISTORY_LOOKBACK_DAYS, missing_days))
        online_df, source_name = fetch_recent_online_history(days=recent_days)
        incremental_mode = True

    online_clean = normalize_repository_dataframe(online_df, source_name)
    if online_clean.empty:
        raise ValueError(f"{source_name}没有返回可写入的数据。")

    previous_count = int(len(local_clean))
    local_issues = set(local_clean["issue"].astype(str).tolist())
    missing_online = online_clean[~online_clean["issue"].astype(str).isin(local_issues)].copy()

    combined = normalize_repository_dataframe(pd.concat([local_clean, online_clean], ignore_index=True), "合并历史数据")
    latest_day = pd.to_datetime(combined["date"], errors="coerce").max()
    if not incremental_mode and not pd.isna(latest_day):
        start_day = latest_day - timedelta(days=days)
        combined_dates = pd.to_datetime(combined["date"], errors="coerce")
        combined = combined[(combined_dates >= start_day) & (combined_dates <= latest_day)].reset_index(drop=True)

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(combined.to_csv(index=False).encode("utf-8-sig"))

    latest_row = combined.iloc[-1] if not combined.empty else pd.Series({"issue": "", "date": "", "number": ""})
    return {
        "path": str(path),
        "source_name": source_name,
        "previous_count": previous_count,
        "row_count": int(len(combined)),
        "added_count": int(len(missing_online)),
        "latest_issue": str(latest_row.get("issue", "")),
        "latest_date": str(latest_row.get("date", "")),
        "latest_number": str(latest_row.get("number", "")),
        "update_mode": "incremental" if incremental_mode else "full",
        "changed": bool(len(missing_online) > 0 or previous_count != len(combined)),
    }


def demo_button_label() -> str:
    action = "重新加载" if demo_reload_required() else "加载"
    return f"{action}{default_history_name()}"


def comparable_metrics(metrics: dict[str, dict[str, float]]) -> dict[str, dict[str, float]]:
    return {
        name: metric
        for name, metric in metrics.items()
        if name != "random_baseline"
    }


def current_generation_metric(metrics: dict[str, dict[str, float]]) -> tuple[str, dict[str, float]] | None:
    usable = comparable_metrics(metrics)
    if not usable:
        return None
    if "ensemble_equal" in usable:
        return "ensemble_equal", usable["ensemble_equal"]
    if "ensemble_weighted" in usable:
        return "ensemble_weighted", usable["ensemble_weighted"]
    return max(usable.items(), key=lambda item: float(item[1].get("top10_hit_rate", 0.0)))


def best_metric(metrics: dict[str, dict[str, float]], key: str) -> tuple[str, dict[str, float]] | None:
    usable = comparable_metrics(metrics)
    if not usable:
        return None
    return max(usable.items(), key=lambda item: float(item[1].get(key, 0.0)))


def baseline_verdict(rate: float, baseline: float, sample_count: int, metric_name: str) -> str:
    if rate > baseline:
        if metric_name == "直选Top10":
            return "直选Top10本次高于随机基线，但不能证明长期有效。"
        if metric_name == "三位7码":
            return "三位7码本次高于随机基线，但仍需更多滚动窗口验证。"
        return f"{metric_name}本次高于随机基线，但仍需更多滚动窗口验证。"
    if rate < baseline:
        return f"{metric_name}未超过随机基线，暂未证明有效。"
    return f"{metric_name}与随机基线接近，暂未证明有效。"


def baseline_comparison_text(metrics: dict[str, dict[str, float]]) -> tuple[str, str]:
    if not metrics:
        return "基线对比结论：等待回测。", "fc3d-warning"

    current = current_generation_metric(metrics)
    best_position = best_metric(metrics, "position_top7_all_hit_rate")
    best_top10 = best_metric(metrics, "top10_hit_rate")
    if current is None or best_position is None or best_top10 is None:
        return "基线对比结论：尚未得到可比较的模型回测结果。", "fc3d-warning"

    current_name, current_metric = current
    current_samples = int(current_metric.get("samples", 0))
    current_position_rate = float(current_metric.get("position_top7_all_hit_rate", 0.0))
    current_top10_rate = float(current_metric.get("top10_hit_rate", 0.0))
    position_verdict = baseline_verdict(current_position_rate, THEORETICAL_POSITION_TOP7_ALL_BASELINE, current_samples, "三位7码")
    top10_verdict = baseline_verdict(current_top10_rate, THEORETICAL_TOP10_BASELINE, current_samples, "直选Top10")

    best_position_name, best_position_metric = best_position
    best_top10_name, best_top10_metric = best_top10
    best_position_rate = float(best_position_metric.get("position_top7_all_hit_rate", 0.0))
    best_top10_rate = float(best_top10_metric.get("top10_hit_rate", 0.0))

    if current_position_rate > THEORETICAL_POSITION_TOP7_ALL_BASELINE and current_top10_rate > THEORETICAL_TOP10_BASELINE and current_samples >= 1000:
        css_class = "fc3d-good"
    elif current_position_rate < THEORETICAL_POSITION_TOP7_ALL_BASELINE and current_top10_rate < THEORETICAL_TOP10_BASELINE:
        css_class = "fc3d-bad"
    else:
        css_class = "fc3d-warning"
    text = (
        f"基线对比结论：{position_verdict}{top10_verdict}"
        f"当前回测参考口径：{display_model_name(current_name)}；候选生成使用保守加权集成排序。"
        f"三位7码最佳模型：{display_model_name(best_position_name)}，命中率 {best_position_rate:.3%}；"
        f"直选Top10最佳模型：{display_model_name(best_top10_name)}，命中率 {best_top10_rate:.3%}。"
    )
    if 0 < current_samples < 1000:
        text += "当前样本不足以证明模型长期有效，建议接入更多历史期数后再比较稳定性。"
    return text, css_class


def css() -> None:
    st.markdown(
        """
        <style>
        :root {
"""
        + theme_css_variables(active_theme_mode())
        + """
        }

        .stApp {
          background: var(--bg);
          color: var(--text);
          font-family: "Segoe UI Variable", "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif;
        }

        .stApp,
        .stApp p,
        .stApp li,
        .stApp label,
        .stApp [data-testid="stMarkdownContainer"],
        .stApp [data-testid="stMarkdownContainer"] p,
        .stApp [data-testid="stWidgetLabel"],
        .stApp [data-testid="stWidgetLabel"] p,
        section[data-testid="stSidebar"],
        section[data-testid="stSidebar"] p,
        section[data-testid="stSidebar"] label,
        section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {
          color: var(--text);
        }

        .stApp small,
        .stApp caption,
        .stApp [data-testid="stCaptionContainer"],
        .stApp [data-testid="stCaptionContainer"] p,
        section[data-testid="stSidebar"] [data-testid="stCaptionContainer"],
        section[data-testid="stSidebar"] [data-testid="stCaptionContainer"] p {
          color: var(--muted) !important;
        }

        .stApp input,
        .stApp textarea,
        .stApp [data-baseweb="input"],
        .stApp [data-baseweb="select"],
        .stApp [data-baseweb="base-input"],
        .stApp [data-testid="stNumberInput"] input {
          background: var(--panel) !important;
          color: var(--text) !important;
          border-color: var(--border) !important;
        }

        .stApp [role="radiogroup"] label,
        .stApp [data-testid="stCheckbox"] label,
        .stApp [data-testid="stCheckbox"] p,
        .stApp [data-testid="stRadio"] label,
        .stApp [data-testid="stRadio"] p {
          color: var(--text) !important;
          opacity: 1 !important;
        }

        .stApp [data-testid="stFileUploaderDropzone"] {
          background: var(--panel-2) !important;
          border: 1px solid var(--border) !important;
        }

        .stApp [data-testid="stFileUploaderDropzone"] * {
          color: var(--text) !important;
          opacity: 1 !important;
        }

        .stApp [data-testid="stAlert"] {
          background: var(--warning-bg) !important;
          color: var(--warning) !important;
          border-color: var(--warning-border) !important;
        }

        header[data-testid="stHeader"],
        [data-testid="stToolbar"],
        #MainMenu,
        footer {
          visibility: hidden;
          height: 0;
        }

        .block-container {
          padding-top: 1.1rem;
          padding-bottom: 1.3rem;
          max-width: 1620px;
        }

        [data-testid="stSidebar"] {
          background: var(--sidebar-bg);
          border-right: 1px solid var(--border);
        }

        header[data-testid="stHeader"] {
          pointer-events: none !important;
        }

        [data-testid="stExpandSidebarButton"] {
          visibility: visible !important;
          pointer-events: auto !important;
          position: fixed !important;
          top: 0.55rem !important;
          left: 0.55rem !important;
          z-index: 3000 !important;
          display: inline-flex !important;
          align-items: center !important;
          justify-content: center !important;
          width: 2rem !important;
          height: 2rem !important;
          background: var(--panel) !important;
          border: 1px solid var(--border) !important;
          border-radius: 8px !important;
          box-shadow: 0 4px 14px var(--shadow) !important;
          color: transparent !important;
          font-size: 0 !important;
          line-height: 0 !important;
          overflow: hidden !important;
        }

        [data-testid="stExpandSidebarButton"] * {
          display: none !important;
          visibility: hidden !important;
          font-size: 0 !important;
          color: transparent !important;
        }

        [data-testid="stExpandSidebarButton"]::before {
          content: "";
          display: block;
          width: 0.9rem;
          height: 2px;
          background: var(--muted);
          border-radius: 999px;
          box-shadow: 0 -5px 0 var(--muted), 0 5px 0 var(--muted);
        }

        .fc3d-title {
          font-size: 2rem;
          font-weight: 700;
          line-height: 1.1;
          letter-spacing: 0;
          color: var(--text);
          margin: 0;
        }

        .fc3d-subtitle {
          color: var(--muted);
          margin-top: 0.3rem;
          font-size: 0.95rem;
        }

        .fc3d-chiprow {
          display: flex;
          gap: 0.5rem;
          flex-wrap: wrap;
          margin-top: 0.75rem;
        }

        .fc3d-chip {
          display: inline-flex;
          align-items: center;
          gap: 0.45rem;
          padding: 0.42rem 0.72rem;
          border: 1px solid var(--border);
          background: var(--panel);
          color: var(--text);
          border-radius: 9px;
          font-size: 0.82rem;
          line-height: 1;
          white-space: nowrap;
        }

        .fc3d-chip b {
          color: var(--accent);
          font-weight: 700;
        }

        .fc3d-panel {
          border: 1px solid var(--border);
          background: var(--panel);
          border-radius: 8px;
          padding: 0.9rem 1rem;
          margin-bottom: 0.8rem;
          box-shadow: 0 8px 20px var(--shadow);
        }

        .fc3d-panel h3, .fc3d-panel h4 {
          margin: 0 0 0.55rem 0;
          color: var(--text);
        }

        .fc3d-note {
          color: var(--muted);
          font-size: 0.86rem;
          line-height: 1.55;
        }

        .fc3d-warning {
          border: 1px solid var(--warning-border);
          background: var(--warning-bg);
          color: var(--warning);
          border-radius: 8px;
          padding: 0.75rem 0.9rem;
        }

        .fc3d-bad {
          border: 1px solid var(--danger);
          background: var(--bad-bg);
          color: var(--bad-text);
          border-radius: 8px;
          padding: 0.75rem 0.9rem;
        }

        .fc3d-good {
          border: 1px solid var(--ok);
          background: var(--good-bg);
          color: var(--good-text);
          border-radius: 8px;
          padding: 0.75rem 0.9rem;
        }

        [data-testid="stExpander"] summary {
          list-style: none;
          position: relative;
          align-items: center;
        }

        [data-testid="stExpander"] summary::before {
          content: "";
          display: inline-block;
          flex: 0 0 auto;
          width: 0;
          height: 0;
          margin-right: 0.35rem;
          border-top: 0.28rem solid transparent;
          border-bottom: 0.28rem solid transparent;
          border-left: 0.42rem solid var(--muted);
          transition: transform 0.16s ease;
        }

        [data-testid="stExpander"] details[open] summary::before {
          transform: rotate(90deg);
        }

        [data-testid="stExpanderToggleIcon"],
        [data-testid="stExpanderToggleIcon"] *,
        [data-testid="stExpander"] summary [data-testid="stIconMaterial"],
        [data-testid="stExpander"] summary [data-testid="stIconMaterial"] *,
        [data-testid="stExpander"] summary span:has(> [data-testid="stIconMaterial"]) {
          display: none !important;
          visibility: hidden !important;
          width: 0 !important;
          height: 0 !important;
          overflow: hidden !important;
          font-size: 0 !important;
          color: transparent !important;
          line-height: 0 !important;
        }

        [data-testid="stFileUploaderDropzoneInstructions"] span {
          font-size: 0 !important;
        }

        [data-testid="stFileUploaderDropzoneInstructions"] span::before {
          content: "把历史开奖CSV拖到这里";
          color: var(--text);
          font-size: 0.86rem;
          font-weight: 700;
        }

        [data-testid="stFileUploaderDropzoneInstructions"] small {
          font-size: 0 !important;
        }

        [data-testid="stFileUploaderDropzoneInstructions"] small::before {
          content: "需要三列：issue期号、date日期、number开奖号";
          color: var(--muted);
          font-size: 0.74rem;
        }

        [data-testid="stFileUploaderDropzone"] button {
          font-size: 0 !important;
        }

        [data-testid="stFileUploaderDropzone"] button::after {
          content: "选择CSV文件";
          font-size: 0.84rem;
          color: inherit;
        }

        [data-testid="stFileUploaderDropzoneInstructions"] {
          position: relative !important;
          min-height: 2.35rem !important;
          justify-content: center !important;
        }

        [data-testid="stFileUploaderDropzoneInstructions"] > * {
          display: none !important;
          visibility: hidden !important;
          font-size: 0 !important;
        }

        [data-testid="stFileUploaderDropzoneInstructions"]::before {
          content: "把历史开奖CSV拖到这里";
          position: absolute;
          top: 0.2rem;
          left: 0;
          right: 0;
          color: var(--text);
          font-size: 0.88rem;
          font-weight: 800;
          text-align: center;
        }

        [data-testid="stFileUploaderDropzoneInstructions"]::after {
          content: "需要三列：issue期号、date日期、number开奖号";
          position: absolute;
          top: 1.35rem;
          left: 0;
          right: 0;
          color: var(--muted);
          font-size: 0.74rem;
          text-align: center;
        }

        [data-testid="stFileUploaderDropzone"] button {
          position: relative !important;
          min-width: 5rem !important;
          color: transparent !important;
          font-size: 0 !important;
          overflow: hidden !important;
          text-indent: -9999px !important;
          white-space: nowrap !important;
        }

        [data-testid="stFileUploaderDropzone"] button * {
          display: none !important;
          visibility: hidden !important;
          font-size: 0 !important;
        }

        [data-testid="stFileUploaderDropzone"] button::before {
          content: "选择CSV文件";
          position: absolute;
          inset: 0;
          display: inline-flex;
          align-items: center;
          justify-content: center;
          color: var(--button-text);
          font-size: 0.84rem;
          font-weight: 700;
          text-indent: 0;
        }

        .fc3d-direct-table-wrap {
          border: 1px solid var(--border);
          border-radius: 8px;
          padding: 0.45rem;
          background: var(--panel);
          overflow-x: auto;
          min-width: 0;
        }

        .fc3d-scroll-hint {
          color: var(--muted);
          font-size: 0.72rem;
          line-height: 1.3;
          margin: 0 0 0.3rem 0;
        }

        .fc3d-direct-table-wrap [data-testid="stDataFrame"] {
          min-width: 320px;
        }

        .fc3d-grid {
          display: grid;
          grid-template-columns: repeat(6, minmax(0, 1fr));
          gap: 0.65rem;
          margin-top: 0.7rem;
        }

        .fc3d-card {
          border: 1px solid var(--border);
          background: var(--panel);
          border-radius: 8px;
          padding: 0.72rem 0.8rem;
          min-height: 74px;
        }

        .fc3d-card .label {
          color: var(--muted);
          font-size: 0.78rem;
          margin-bottom: 0.25rem;
        }

        .fc3d-card .value {
          color: var(--text);
          font-size: 1.15rem;
          font-weight: 700;
          line-height: 1.15;
          word-break: break-word;
        }

        .fc3d-compact {
          display: grid;
          grid-template-columns: repeat(2, minmax(0, 1fr));
          gap: 0.55rem;
        }

        .fc3d-topbar-grid {
          display: grid;
          grid-template-columns: repeat(6, minmax(0, 1fr));
          gap: 0.55rem;
          margin-top: 0.7rem;
        }

        .fc3d-mini {
          border: 1px solid var(--border);
          background: var(--panel);
          border-radius: 8px;
          padding: 0.7rem 0.75rem;
        }

        .fc3d-mini .label {
          color: var(--muted);
          font-size: 0.78rem;
          margin-bottom: 0.2rem;
        }

        .fc3d-mini .value {
          color: var(--text);
          font-weight: 700;
          font-size: 0.98rem;
          word-break: break-word;
        }

        .fc3d-candidate-cards {
          display: grid;
          grid-template-columns: repeat(3, minmax(0, 1fr));
          gap: 0.7rem;
          margin: 0.75rem 0;
        }

        .fc3d-candidate-card {
          border: 1px solid var(--border);
          background: var(--panel);
          border-radius: 8px;
          padding: 0.82rem;
          box-shadow: 0 6px 18px var(--shadow);
        }

        .fc3d-candidate-card .rank {
          color: var(--muted);
          font-size: 0.75rem;
          font-weight: 700;
        }

        .fc3d-candidate-card .number {
          color: var(--text);
          font-size: 1.55rem;
          font-weight: 800;
          letter-spacing: 0;
          line-height: 1.15;
        }

        .fc3d-candidate-card .score {
          color: var(--accent);
          font-size: 0.86rem;
          font-weight: 700;
          margin-top: 0.2rem;
        }

        .fc3d-candidate-card .models {
          color: var(--muted);
          font-size: 0.78rem;
          margin-top: 0.25rem;
          line-height: 1.4;
        }

        .fc3d-candidate-card.fc3d-extra-card {
          display: none;
        }

        .fc3d-compact-candidate-grid {
          display: grid;
          grid-template-columns: repeat(5, minmax(0, 1fr));
          gap: 0.48rem;
          margin: 0.65rem 0 0.75rem 0;
        }

        .fc3d-compact-candidate {
          min-width: 0;
          border: 1px solid var(--border);
          background: var(--panel);
          border-radius: 8px;
          padding: 0.52rem 0.58rem;
          box-shadow: 0 4px 12px var(--shadow);
        }

        .fc3d-compact-candidate .rank {
          color: var(--muted);
          font-size: 0.68rem;
          font-weight: 700;
        }

        .fc3d-compact-candidate .number {
          color: var(--text);
          font-size: 1.28rem;
          font-weight: 800;
          letter-spacing: 0;
          line-height: 1.1;
        }

        .fc3d-compact-candidate .score {
          color: var(--accent);
          font-size: 0.72rem;
          font-weight: 700;
          margin-top: 0.12rem;
          line-height: 1.2;
        }

        .fc3d-compact-candidate .models {
          color: var(--muted);
          font-size: 0.68rem;
          margin-top: 0.15rem;
          line-height: 1.25;
          overflow-wrap: anywhere;
        }

        .fc3d-candidate-list {
          display: grid;
          grid-template-columns: repeat(2, minmax(0, 1fr));
          gap: 0.55rem;
          margin-top: 0.55rem;
        }

        .fc3d-candidate-row {
          display: grid;
          grid-template-columns: 2.2rem 4.2rem minmax(0, 1fr);
          gap: 0.5rem;
          align-items: center;
          border: 1px solid var(--border);
          background: var(--panel);
          border-radius: 8px;
          padding: 0.56rem 0.62rem;
        }

        .fc3d-candidate-row .rank {
          color: var(--muted);
          font-size: 0.74rem;
          font-weight: 700;
        }

        .fc3d-candidate-row .number {
          color: var(--text);
          font-size: 1.3rem;
          font-weight: 800;
          letter-spacing: 0;
          line-height: 1;
        }

        .fc3d-candidate-row .meta {
          min-width: 0;
          color: var(--muted);
          font-size: 0.76rem;
          line-height: 1.35;
        }

        .fc3d-candidate-row .meta b {
          color: var(--accent);
        }

        .fc3d-empty-candidates {
          border: 1px dashed var(--border);
          background: var(--panel-2);
          color: var(--muted);
          border-radius: 8px;
          padding: 0.75rem;
          margin-top: 0.5rem;
          text-align: center;
        }

        .fc3d-weight-summary {
          display: grid;
          grid-template-columns: repeat(3, minmax(0, 1fr));
          gap: 0.55rem;
          margin-bottom: 0.65rem;
        }

        .fc3d-weight-pill {
          border: 1px solid #BFDBFE;
          background: var(--secondary-bg);
          border-radius: 8px;
          padding: 0.62rem 0.65rem;
        }

        .fc3d-weight-pill .label {
          color: var(--secondary-text);
          font-size: 0.74rem;
          margin-bottom: 0.18rem;
        }

        .fc3d-weight-pill .value {
          color: var(--text);
          font-size: 1rem;
          font-weight: 800;
          line-height: 1.1;
        }

        .fc3d-position-grid {
          display: grid;
          grid-template-columns: repeat(3, minmax(0, 1fr));
          gap: 0.7rem;
          margin-top: 0.7rem;
        }

        .fc3d-position-source {
          display: grid;
          grid-template-columns: repeat(2, minmax(0, 1fr));
          gap: 0.55rem;
          margin: 0.65rem 0;
        }

        .fc3d-position-source .fc3d-mini {
          background: var(--panel-2);
          box-shadow: none;
        }

        .fc3d-position-card {
          border: 1px solid var(--border);
          background: var(--panel);
          border-radius: 8px;
          padding: 0.75rem;
        }

        .fc3d-position-card .title {
          color: var(--muted);
          font-size: 0.78rem;
          margin-bottom: 0.45rem;
        }

        .fc3d-digit-row {
          display: flex;
          flex-wrap: wrap;
          gap: 0.35rem;
        }

        .fc3d-digit-pill {
          min-width: 2.35rem;
          border-radius: 999px;
          background: var(--secondary-bg);
          color: var(--secondary-text);
          border: 1px solid #C7D2FE;
          padding: 0.22rem 0.42rem;
          text-align: center;
          font-weight: 800;
        }

        .fc3d-digit-pill small {
          display: block;
          font-size: 0.62rem;
          font-weight: 600;
          color: var(--muted);
          line-height: 1.1;
        }

        .fc3d-export-note {
          border: 1px solid #CBD5E1;
          background: var(--panel-2);
          border-radius: 8px;
          padding: 0.7rem 0.75rem;
          margin: 0.65rem 0;
          color: var(--text);
          font-size: 0.86rem;
        }

        .stDataFrame, .stTable {
          border: 1px solid var(--border);
          border-radius: 8px;
          overflow: hidden;
        }

        [data-testid="stMetric"] {
          background: var(--panel);
          border: 1px solid var(--border);
          border-radius: 8px;
          padding: 0.65rem 0.75rem;
        }

        [data-testid="stMetric"] [data-testid="stMetricLabel"] p {
          color: var(--muted);
          font-size: 0.8rem;
        }

        [data-testid="stMetric"] [data-testid="stMetricValue"] {
          color: var(--text);
          font-weight: 700;
        }

        section[data-testid="stSidebar"] .stButton button,
        section[data-testid="stSidebar"] .stDownloadButton button {
          width: 100%;
        }

        .stButton button,
        .stDownloadButton button {
          border-radius: 8px !important;
          border: 1px solid var(--border) !important;
          background: var(--button-bg) !important;
          color: var(--button-text) !important;
          font-weight: 700 !important;
          box-shadow: none !important;
        }

        div.st-key-main_validate_data button:not(:disabled),
        div.st-key-sidebar_validate_data button:not(:disabled) {
          background: var(--validate-bg) !important;
          border-color: var(--accent) !important;
          color: var(--validate-text) !important;
        }

        .stButton button:disabled,
        .stDownloadButton button:disabled {
          background: var(--disabled-bg) !important;
          color: var(--disabled-text) !important;
          border-color: var(--border) !important;
          opacity: 1 !important;
        }

        .stButton button:disabled *,
        .stDownloadButton button:disabled * {
          color: var(--disabled-text) !important;
          opacity: 1 !important;
        }

        .stButton button[kind="primary"],
        .stDownloadButton button[kind="primary"] {
          background: var(--accent) !important;
          border-color: var(--accent) !important;
          color: var(--primary-button-text) !important;
        }

        div.st-key-candidate_export_report button:not(:disabled),
        div.st-key-export_full_report button:not(:disabled) {
          background: var(--solid-button-bg) !important;
          border-color: var(--solid-button-bg) !important;
          color: var(--solid-button-text) !important;
        }

        .stButton button:disabled,
        .stDownloadButton button:disabled,
        .stButton button[kind="primary"]:disabled {
          background: var(--disabled-bg) !important;
          color: var(--disabled-text) !important;
          border-color: var(--border) !important;
          opacity: 1 !important;
        }

        div.st-key-main_action_panel {
          border: 1px solid var(--border);
          background: var(--panel);
          border-radius: 8px;
          padding: 0.9rem 1rem;
          margin-bottom: 0.8rem;
          box-shadow: 0 8px 20px var(--shadow);
        }

        div.st-key-main_action_panel h3 {
          margin: 0 0 0.45rem 0;
          color: var(--text);
        }

        .fc3d-action-grid {
          display: grid;
          grid-template-columns: minmax(0, 1.3fr) minmax(0, 1fr);
          gap: 0.55rem;
          margin: 0.45rem 0 0.7rem 0;
        }

        .fc3d-next-action,
        .fc3d-disabled-reason {
          border: 1px solid var(--border);
          background: var(--panel-2);
          border-radius: 8px;
          padding: 0.62rem 0.7rem;
          color: var(--muted);
          font-size: 0.82rem;
          line-height: 1.45;
        }

        .fc3d-next-action span,
        .fc3d-disabled-reason span {
          display: block;
          color: var(--muted);
          font-size: 0.72rem;
          margin-bottom: 0.12rem;
        }

        .fc3d-next-action b {
          color: var(--accent);
          font-size: 1rem;
        }

        div.st-key-main_action_panel [data-testid="stHorizontalBlock"] {
          gap: 0.55rem !important;
          align-items: stretch !important;
        }

        div.st-key-main_action_panel [data-testid="column"] {
          min-width: 8rem !important;
        }

        div.st-key-main_action_panel .stButton button {
          width: 100%;
          min-height: 2.35rem;
          white-space: normal;
          line-height: 1.25;
        }

        .fc3d-backtest-conclusion-grid {
          display: grid;
          grid-template-columns: repeat(6, minmax(0, 1fr));
          gap: 0.5rem;
          margin: 0.55rem 0 0.65rem 0;
        }

        .fc3d-conclusion-card {
          border: 1px solid var(--border);
          background: var(--panel-2);
          border-radius: 8px;
          padding: 0.64rem 0.68rem;
          min-width: 0;
        }

        .fc3d-conclusion-card .label {
          color: var(--muted);
          font-size: 0.72rem;
          margin-bottom: 0.18rem;
        }

        .fc3d-conclusion-card .value {
          color: var(--text);
          font-weight: 800;
          font-size: 0.92rem;
          line-height: 1.25;
          overflow-wrap: anywhere;
        }

        .fc3d-sig-badge,
        .fc3d-sig-yes,
        .fc3d-sig-no {
          display: inline-flex;
          align-items: center;
          justify-content: center;
          border-radius: 999px;
          padding: 0.14rem 0.48rem;
          font-size: 0.72rem;
          font-weight: 800;
          line-height: 1.2;
        }

        .fc3d-sig-yes {
          background: var(--good-bg);
          color: var(--good-text);
          border: 1px solid var(--ok);
        }

        .fc3d-sig-no {
          background: var(--disabled-bg);
          color: var(--disabled-text);
          border: 1px solid var(--border);
        }

        .fc3d-danma-grid {
          display: grid;
          grid-template-columns: repeat(3, minmax(0, 1fr));
          gap: 0.7rem;
          margin: 0.75rem 0;
        }

        .fc3d-danma-card {
          border: 1px solid var(--border);
          background: var(--panel-2);
          border-radius: 8px;
          padding: 0.78rem;
          min-width: 0;
        }

        .fc3d-danma-card .digit {
          color: var(--accent);
          font-size: 2rem;
          font-weight: 900;
          line-height: 1;
        }

        .fc3d-danma-card .meta {
          color: var(--muted);
          font-size: 0.78rem;
          line-height: 1.35;
          margin-top: 0.35rem;
          overflow-wrap: anywhere;
        }

        .fc3d-candidate-section {
          border-top: 3px solid var(--accent);
          padding-top: 0.55rem;
          margin-top: 0.75rem;
        }

        .fc3d-candidate-section.diverse {
          border-top-color: var(--accent-2);
        }

        .fc3d-candidate-section.long-tail {
          border-top-color: var(--warning-border);
        }

        .fc3d-export-center-grid {
          display: grid;
          grid-template-columns: repeat(2, minmax(0, 1fr));
          gap: 0.65rem;
          margin: 0.65rem 0;
        }

        [data-testid="stCodeBlock"] pre {
          background: var(--code-bg) !important;
          color: var(--code-text) !important;
          border-radius: 8px !important;
        }

        [data-testid="stFileUploaderDropzoneInstructions"] > span {
          display: none !important;
          visibility: hidden !important;
        }

        [data-testid="stFileUploaderDropzoneInstructions"] > div {
          display: flex !important;
          flex-direction: column !important;
          gap: 0.16rem !important;
        }

        [data-testid="stFileUploaderDropzoneInstructions"] > div span {
          color: transparent !important;
          font-size: 0 !important;
          line-height: 1.4 !important;
        }

        [data-testid="stFileUploaderDropzoneInstructions"] > div span:first-child::before {
          content: "把历史开奖CSV拖到这里";
          font-size: 0.88rem;
          color: var(--text);
        }

        [data-testid="stFileUploaderDropzoneInstructions"] > div span:nth-child(2)::before {
          content: "需要三列：issue期号、date日期、number开奖号";
          font-size: 0.78rem;
          color: var(--muted);
        }

        [data-testid="stFileUploaderDropzone"] button {
          position: relative !important;
          color: transparent !important;
          font-size: 0 !important;
          overflow: hidden !important;
          text-indent: -9999px !important;
          white-space: nowrap !important;
        }

        [data-testid="stFileUploaderDropzone"] button::after {
          content: "" !important;
          display: none !important;
        }

        [data-testid="stFileUploaderDropzone"] button::before {
          content: "选择CSV文件";
          position: absolute;
          inset: 0;
          display: inline-flex;
          align-items: center;
          justify-content: center;
          font-size: 0.88rem;
          color: var(--secondary-text);
          text-indent: 0;
        }

        .fc3d-nav-note {
          margin: 0.35rem 0 0.75rem 0;
        }

        @media (max-width: 700px) {
          [data-testid="stExpandSidebarButton"] {
            visibility: visible !important;
            pointer-events: auto !important;
            position: fixed !important;
            top: 0.55rem !important;
            left: 0.55rem !important;
            z-index: 3000 !important;
            background: var(--panel) !important;
            border: 1px solid var(--border) !important;
            border-radius: 8px !important;
            box-shadow: 0 4px 14px var(--shadow) !important;
          }

          [data-testid="stExpandSidebarButton"] * {
            visibility: visible !important;
          }

          header[data-testid="stHeader"] {
            pointer-events: none !important;
          }

          header[data-testid="stHeader"] [data-testid="stExpandSidebarButton"] {
            pointer-events: auto !important;
          }

          .block-container {
            padding-left: 0.55rem;
            padding-right: 0.55rem;
            padding-top: 0.75rem;
          }

          .fc3d-title {
            font-size: 1.35rem;
            line-height: 1.18;
          }

          .fc3d-subtitle,
          .fc3d-note,
          .fc3d-chip {
            font-size: 0.82rem;
          }

          .fc3d-panel {
            padding: 0.72rem 0.68rem;
            border-radius: 8px;
          }

          div.st-key-main_action_panel {
            padding: 0.72rem 0.68rem;
            border-radius: 8px;
          }

          .fc3d-chip {
            white-space: normal;
            align-items: flex-start;
          }

          .fc3d-chiprow {
            display: none;
          }

          .fc3d-topbar-grid {
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 0.45rem;
            margin-top: 0.55rem;
          }

          .fc3d-topbar-grid .fc3d-mini {
            padding: 0.55rem 0.6rem;
          }

          .fc3d-topbar-grid .fc3d-mini .label {
            font-size: 0.72rem;
          }

          .fc3d-topbar-grid .fc3d-mini .value {
            font-size: 0.88rem;
            line-height: 1.2;
          }

          .fc3d-grid {
            grid-template-columns: 1fr;
          }

          .fc3d-action-grid,
          .fc3d-backtest-conclusion-grid,
          .fc3d-export-center-grid {
            grid-template-columns: 1fr;
          }

          .fc3d-candidate-cards {
            grid-template-columns: 1fr;
          }

          .fc3d-compact-candidate-grid {
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 0.42rem;
          }

          .fc3d-compact-candidate {
            padding: 0.5rem 0.52rem;
          }

          .fc3d-compact-candidate .number {
            font-size: 1.14rem;
          }

          .fc3d-candidate-list {
            grid-template-columns: 1fr;
          }

          .fc3d-candidate-row {
            grid-template-columns: 1.8rem 3.6rem minmax(0, 1fr);
            gap: 0.42rem;
            padding: 0.52rem 0.55rem;
          }

          .fc3d-candidate-row .number {
            font-size: 1.15rem;
          }

          .fc3d-candidate-card.fc3d-extra-card {
            display: block;
          }

          .fc3d-mobile-hide,
          div.st-key-candidate_wide_table {
            display: none !important;
          }

          .fc3d-weight-summary {
            grid-template-columns: 1fr;
          }

          .fc3d-danma-grid,
          .fc3d-position-grid {
            grid-template-columns: 1fr;
          }

          .fc3d-position-source {
            grid-template-columns: 1fr;
          }

          .fc3d-card,
          .fc3d-mini {
            min-width: 0;
          }

          .fc3d-card .value,
          .fc3d-mini .value,
          .fc3d-chip,
          .fc3d-note,
          p,
          span,
          div {
            overflow-wrap: anywhere;
            word-break: normal;
          }

          [data-testid="stHorizontalBlock"] {
            flex-wrap: wrap !important;
            gap: 0.55rem !important;
          }

          [data-testid="column"] {
            flex: 1 1 100% !important;
            min-width: 100% !important;
            width: 100% !important;
          }

          section[data-testid="stSidebar"][aria-expanded="false"] {
            width: 0 !important;
            min-width: 0 !important;
            max-width: 0 !important;
          }

          section[data-testid="stSidebar"][aria-expanded="false"] [data-testid="stSidebarContent"] {
            display: none !important;
          }

          section[data-testid="stSidebar"][aria-expanded="true"] {
            position: fixed !important;
            inset: 0 auto 0 0 !important;
            width: min(88vw, 320px) !important;
            max-width: 320px !important;
            z-index: 2500 !important;
            box-shadow: 18px 0 40px var(--shadow);
          }

          section[data-testid="stSidebar"][aria-expanded="true"] + div,
          section[data-testid="stSidebar"][aria-expanded="true"] ~ div {
            margin-left: 0 !important;
          }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def load_default_history() -> tuple[pd.DataFrame, str, str]:
    if BUILTIN_HISTORY_PATH.exists():
        df = read_csv_bytes(BUILTIN_HISTORY_PATH.read_bytes())
        return df, f"{BUILTIN_HISTORY_PATH.name}（本地缓存）", "近5年开奖记录"
    df, source_name = fetch_online_history()
    BUILTIN_HISTORY_PATH.write_bytes(df.to_csv(index=False).encode("utf-8-sig"))
    return df, f"{BUILTIN_HISTORY_PATH.name}（{source_name}更新）", "近5年开奖记录"


def read_csv_bytes(data: bytes) -> pd.DataFrame:
    encodings = ["utf-8-sig", "utf-8", "gb18030", "gbk"]
    last_error: Exception | None = None
    for enc in encodings:
        try:
            return pd.read_csv(io.BytesIO(data), dtype="string", keep_default_na=False, encoding=enc)
        except Exception as exc:
            last_error = exc
    raise ValueError(f"无法读取 CSV，已尝试编码 {encodings}，最后错误：{last_error}")


def inspect_and_clean(raw_df: pd.DataFrame, file_name: str, raw_bytes: bytes) -> tuple[pd.DataFrame, dict[str, Any]]:
    core = get_core()
    df = raw_df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]

    required_cols = ["issue", "date", "number"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"CSV 缺少必要字段：{missing_cols}")

    df = df[required_cols].copy()
    for col in required_cols:
        df[col] = df[col].astype("string").str.strip()
        df.loc[df[col] == "", col] = pd.NA

    raw_rows = len(df)
    missing_mask = df[required_cols].isna().any(axis=1)
    missing_rows = df.loc[missing_mask].copy()
    df = df.loc[~missing_mask].copy()

    invalid_issue_mask = df["issue"].isna() | ~df["issue"].str.fullmatch(r"\d{5,8}").fillna(False)
    invalid_issue_rows = df.loc[invalid_issue_mask].copy()
    df = df.loc[~invalid_issue_mask].copy()

    invalid_number_mask = df["number"].isna() | ~df["number"].str.fullmatch(r"\d{3}").fillna(False)
    invalid_number_rows = df.loc[invalid_number_mask].copy()
    df = df.loc[~invalid_number_mask].copy()

    parsed_dates = pd.to_datetime(df["date"], errors="coerce")
    invalid_date_mask = parsed_dates.isna()
    invalid_date_rows = df.loc[invalid_date_mask].copy()
    df = df.loc[~invalid_date_mask].copy()
    parsed_dates = parsed_dates.loc[~invalid_date_mask]
    date_order_issue_count = int((parsed_dates.diff().dt.days < 0).sum()) if len(parsed_dates) else 0

    duplicate_issue_rows = df.loc[df.duplicated("issue", keep=False)].copy()
    df = df.drop_duplicates("issue", keep="first").copy()

    df["_date"] = parsed_dates.loc[df.index].values if len(df) == len(parsed_dates) else pd.to_datetime(df["date"], errors="coerce")
    df = df.sort_values(["_date", "issue"]).drop(columns=["_date"]).reset_index(drop=True)
    clean_df = core.add_number_columns(df)

    leading_zero_count = int(clean_df["number"].str.startswith("0").sum())
    data_version = short_hash(raw_bytes)
    data_hash_sha256 = full_hash(raw_bytes)
    sorted_dates = pd.to_datetime(clean_df["date"], errors="coerce") if not clean_df.empty else pd.Series(dtype="datetime64[ns]")
    gap_days: list[str] = []
    if len(sorted_dates) > 1:
        for previous, current in zip(sorted_dates.iloc[:-1], sorted_dates.iloc[1:]):
            day_gap = int((current - previous).days)
            if day_gap > 1:
                gap_days.extend(
                    (previous + timedelta(days=offset)).strftime("%Y-%m-%d")
                    for offset in range(1, day_gap)
                )
    sample_warning = "当前样本量较少，模型结果仅适合学习和模拟研究。"
    report: dict[str, Any] = {
        "file_name": file_name,
        "data_version": data_version,
        "data_hash_sha256": data_hash_sha256,
        "raw_rows": raw_rows,
        "kept_rows": int(len(clean_df)),
        "dropped_rows": int(raw_rows - len(clean_df)),
        "missing_rows_count": int(len(missing_rows)),
        "invalid_issue_count": int(len(invalid_issue_rows)),
        "invalid_number_count": int(len(invalid_number_rows)),
        "invalid_date_count": int(len(invalid_date_rows)),
        "duplicate_issue_count": int(duplicate_issue_rows["issue"].nunique()),
        "leading_zero_count": leading_zero_count,
        "date_order_issue_count": date_order_issue_count,
        "calendar_gap_days_count": int(len(gap_days)),
        "calendar_gap_days": gap_days[:40],
        "date_min": clean_df["date"].iloc[0] if not clean_df.empty else "",
        "date_max": clean_df["date"].iloc[-1] if not clean_df.empty else "",
        "missing_rows": missing_rows.head(20).to_dict("records"),
        "invalid_issue_rows": invalid_issue_rows.head(20).to_dict("records"),
        "invalid_number_rows": invalid_number_rows.head(20).to_dict("records"),
        "invalid_date_rows": invalid_date_rows.head(20).to_dict("records"),
        "duplicate_issue_rows": duplicate_issue_rows.head(20).to_dict("records"),
        "clean_preview": clean_df.head(6).to_dict("records"),
        "warning_small_sample": len(clean_df) < 1000,
        "sample_warning_text": sample_warning,
    }
    return clean_df, report


def format_top_models(model_scores: dict[str, np.ndarray], idx: int) -> str:
    core = get_core()
    items = sorted(
        [
            (name, float(core.calibrate_model_scores(scores)[idx]))
            for name, scores in model_scores.items()
            if name != "random_baseline"
        ],
        key=lambda x: x[1],
        reverse=True,
    )
    return " / ".join(display_model_name(name) for name, _value in items[:3])


def active_score_items(model_scores: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {
        name: scores
        for name, scores in model_scores.items()
        if name != "random_baseline" and not name.startswith("__")
    }


def normalize_digit_component(values: np.ndarray | list[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.size != 10:
        arr = np.resize(arr, 10).astype(float)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    low = float(np.min(arr))
    high = float(np.max(arr))
    if high - low <= 1e-12:
        return np.zeros(10, dtype=float)
    return (arr - low) / (high - low)


def raw_digit_frequency(history: pd.DataFrame, window: int | None = None) -> np.ndarray:
    if history is None or history.empty:
        return np.zeros(10, dtype=float)
    hist = history.tail(window).copy() if window else history.copy()
    if hist.empty:
        return np.zeros(10, dtype=float)
    hist = hist if "bai" in hist.columns else get_core().add_number_columns(hist)
    digits = hist[["bai", "shi", "ge"]].to_numpy(dtype=int).reshape(-1)
    counts = np.bincount(digits, minlength=10).astype(float)
    total = float(counts.sum())
    return counts / total if total > 0 else np.zeros(10, dtype=float)


def digit_code_score_components(scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    core = get_core()
    calibrated = core.normalize_scores(np.asarray(scores, dtype=float))
    marginals = core.position_marginals_from_code_scores(calibrated)
    position_marginal = normalize_digit_component((marginals[0] + marginals[1] + marginals[2]) / 3.0)
    unique_presence = np.zeros(10, dtype=float)
    code_digits = np.asarray(core.CODE_DIGITS, dtype=int)
    for digit in range(10):
        unique_presence[digit] = float(calibrated[np.any(code_digits == digit, axis=1)].sum())
    return position_marginal, normalize_digit_component(unique_presence)


def digit_position_frequency(history: pd.DataFrame) -> np.ndarray:
    if history is None or history.empty:
        return np.zeros(10, dtype=float)
    hist = history if "bai" in history.columns else get_core().add_number_columns(history)
    parts: list[np.ndarray] = []
    for pos in ["bai", "shi", "ge"]:
        counts = np.bincount(hist[pos].to_numpy(dtype=int), minlength=10).astype(float)
        total = float(counts.sum())
        parts.append(counts / total if total > 0 else np.zeros(10, dtype=float))
    return normalize_digit_component(np.mean(parts, axis=0))


def digit_omission_rebound(history: pd.DataFrame) -> np.ndarray:
    if history is None or history.empty:
        return np.zeros(10, dtype=float)
    hist = history if "bai" in history.columns else get_core().add_number_columns(history)
    last_seen = {digit: None for digit in range(10)}
    for idx, row in enumerate(hist[["bai", "shi", "ge"]].to_numpy(dtype=int)):
        for digit in set(int(value) for value in row):
            last_seen[digit] = idx
    n = len(hist)
    gaps = np.array(
        [n if last_seen[digit] is None else max(0, n - 1 - int(last_seen[digit])) for digit in range(10)],
        dtype=float,
    )
    return normalize_digit_component(np.log1p(gaps))


def digit_sum_distribution_support(history: pd.DataFrame) -> np.ndarray:
    core = get_core()
    if history is None or history.empty:
        return np.zeros(10, dtype=float)
    hist = history if "sum3" in history.columns else core.add_number_columns(history)
    sum_counts = np.bincount(hist["sum3"].to_numpy(dtype=int), minlength=28).astype(float)
    sum_probs = (sum_counts + 1.0) / (float(len(hist)) + 28.0)
    code_digits = np.asarray(core.CODE_DIGITS, dtype=int)
    code_sums = np.asarray(core.CODE_SUMS, dtype=int)
    support = np.zeros(10, dtype=float)
    for digit in range(10):
        mask = np.any(code_digits == digit, axis=1)
        support[digit] = float(np.mean(sum_probs[code_sums[mask]])) if np.any(mask) else 0.0
    return normalize_digit_component(support)


def digit_heat_cold_stability(history: pd.DataFrame) -> np.ndarray:
    recent = raw_digit_frequency(history, 30)
    longer = raw_digit_frequency(history, 100)
    recent_norm = normalize_digit_component(recent)
    longer_norm = normalize_digit_component(longer)
    stability = 1.0 - np.abs(recent_norm - longer_norm)
    return normalize_digit_component(stability)


def build_danma_history_components(history: pd.DataFrame) -> dict[str, np.ndarray]:
    return {
        "near30_frequency": normalize_digit_component(raw_digit_frequency(history, 30)),
        "near100_frequency": normalize_digit_component(raw_digit_frequency(history, 100)),
        "position_frequency": digit_position_frequency(history),
        "omission_rebound": digit_omission_rebound(history),
        "sum_distribution_support": digit_sum_distribution_support(history),
        "heat_cold_stability": digit_heat_cold_stability(history),
    }


def build_digit_score_components(scores: np.ndarray, history: pd.DataFrame) -> dict[str, np.ndarray]:
    position_marginal, unique_presence = digit_code_score_components(scores)
    components: dict[str, np.ndarray] = {
        "position_marginal_score": position_marginal,
        "unique_presence_score": unique_presence,
        "combination_marginal_score": normalize_digit_component((position_marginal + unique_presence) / 2.0),
    }
    components.update(build_danma_history_components(history))
    return components


def top_digit_source_models(model_scores: dict[str, np.ndarray], digit: int, limit: int = 3) -> list[str]:
    items: list[tuple[str, float]] = []
    for name, scores in active_score_items(model_scores or {}).items():
        position_score, unique_score = digit_code_score_components(scores)
        digit_score = float((position_score[int(digit)] + unique_score[int(digit)]) / 2.0)
        items.append((name, digit_score))
    items.sort(key=lambda item: item[1], reverse=True)
    return [display_model_name(name) for name, _value in items[:limit]]


def build_danma_ranking(
    model_scores: dict[str, np.ndarray],
    components: dict[str, np.ndarray],
    include_sources: bool = True,
    score_weights: dict[str, float] | None = None,
) -> pd.DataFrame:
    weights = score_weights or DANMA_SCORE_WEIGHTS
    total_score = np.zeros(10, dtype=float)
    for key, weight in weights.items():
        total_score += components[key] * float(weight)
    total_score = normalize_digit_component(total_score)

    rows: list[dict[str, Any]] = []
    for digit in range(10):
        contribution_items = sorted(
            [
                (key, float(components[key][digit]) * float(weight))
                for key, weight in weights.items()
            ],
            key=lambda item: item[1],
            reverse=True,
        )
        unique_support: list[str] = []
        if include_sources:
            support_labels: list[str] = []
            for key, _value in contribution_items:
                if key == "combination_marginal_score":
                    support_labels.extend(top_digit_source_models(model_scores, digit, limit=2))
                else:
                    support_labels.append(DANMA_COMPONENT_LABELS.get(key, key))
                if len(dict.fromkeys(support_labels)) >= 3:
                    break
            unique_support = list(dict.fromkeys(label for label in support_labels if label))
        row = {
            "digit": digit,
            "score": float(total_score[digit]),
            "source_models": " / ".join(unique_support[:3]) if unique_support else "集成排序",
        }
        for key, values in components.items():
            row[key] = float(values[digit])
        rows.append(row)

    ranking = pd.DataFrame(rows).sort_values(["score", "digit"], ascending=[False, True]).reset_index(drop=True)
    ranking.insert(0, "rank", np.arange(1, len(ranking) + 1))
    return ranking


def build_danma_prediction(
    clean_df: pd.DataFrame,
    model_scores: dict[str, np.ndarray],
    ensemble_scores: np.ndarray,
    data_report: dict[str, Any] | None = None,
    generated_at: str | None = None,
) -> dict[str, Any]:
    core = get_core()
    hist = clean_df if "bai" in clean_df.columns else core.add_number_columns(clean_df)
    confidence_rule = get_active_danma_rule()
    weights = dict(confidence_rule.get("weights", DANMA_SCORE_WEIGHTS))
    source_model = str(confidence_rule.get("model", "ensemble_equal"))
    source_scores = active_score_items(model_scores or {}).get(source_model, ensemble_scores)
    components = build_digit_score_components(source_scores, hist)
    ranking = build_danma_ranking(model_scores, components, include_sources=True, score_weights=weights)
    digits = [int(value) for value in ranking.head(3)["digit"].tolist()]
    data_report = data_report or {}
    latest_issue = str(hist.iloc[-1].get("issue", "")) if not hist.empty else ""
    latest_date = str(data_report.get("date_max") or (hist.iloc[-1].get("date", "") if not hist.empty else ""))
    return {
        "digits": digits,
        "ranking": ranking,
        "generated_at": generated_at or now_str(),
        "data_version": str(data_report.get("data_version", "")),
        "latest_issue": latest_issue,
        "latest_date": latest_date,
        "score_note": DANMA_SCORE_NOTE,
        "weights": dict(weights),
        "confidence_threshold": build_confidence_threshold_status(ranking, confidence_rule, pool_size=3),
    }


def digit_ranking_confidence_value(ranking: pd.DataFrame, feature: str, pool_size: int) -> float:
    if ranking is None or ranking.empty:
        return 0.0
    scores = ranking.sort_values("rank")["score"].to_numpy(dtype=float)
    if len(scores) < pool_size:
        return 0.0
    top = scores[:pool_size]
    bottom = scores[pool_size:]
    if feature == "gap" and len(scores) > pool_size:
        return float(scores[pool_size - 1] - scores[pool_size])
    if feature == "spread" and len(bottom) > 0:
        return float(np.mean(top) - np.mean(bottom))
    if feature == "top_min":
        return float(scores[pool_size - 1])
    if feature == "top_sum":
        return float(np.sum(top))
    if feature == "bottom_sum":
        return float(np.sum(bottom)) if len(bottom) > 0 else 0.0
    return 0.0


def evaluate_confidence_condition(ranking: pd.DataFrame, condition: dict[str, Any], pool_size: int) -> dict[str, Any]:
    feature = str(condition.get("feature", ""))
    direction = str(condition.get("direction", ""))
    threshold = condition.get("threshold")
    value = digit_ranking_confidence_value(ranking, feature, pool_size)
    if threshold is None or direction == "all":
        passes = True
    elif direction == "<=":
        passes = value <= float(threshold)
    elif direction == ">=":
        passes = value >= float(threshold)
    else:
        passes = False
    return {
        "feature": feature,
        "direction": direction,
        "threshold": threshold,
        "value": float(value),
        "passes": bool(passes),
    }


def build_confidence_threshold_status(ranking: pd.DataFrame, rule: dict[str, Any], pool_size: int) -> dict[str, Any]:
    raw_conditions = rule.get("conditions") or []
    if raw_conditions:
        conditions = [evaluate_confidence_condition(ranking, condition, pool_size) for condition in raw_conditions]
        first = conditions[0]
        feature = str(first.get("feature", ""))
        direction = str(first.get("direction", ""))
        threshold = first.get("threshold")
        value = float(first.get("value", 0.0))
        passes = all(bool(condition.get("passes")) for condition in conditions)
    else:
        conditions = []
        evaluated = evaluate_confidence_condition(ranking, rule, pool_size)
        feature = str(evaluated.get("feature", ""))
        direction = str(evaluated.get("direction", ""))
        threshold = evaluated.get("threshold")
        value = float(evaluated.get("value", 0.0))
        passes = bool(evaluated.get("passes"))
    return {
        "model": str(rule.get("model", "")),
        "feature": feature,
        "direction": direction,
        "threshold": threshold,
        "value": float(value),
        "passes": bool(passes),
        "conditions": conditions,
        "target_rate": float(rule.get("target_rate", 0.0)),
        "validation_rate": float(rule.get("validation_rate", 0.0)),
        "validation_hits": int(rule.get("validation_hits", 0)),
        "validation_selected": int(rule.get("validation_selected", 0)),
        "validation_samples": int(rule.get("validation_samples", 0)),
        "tune_rate": float(rule.get("tune_rate", 0.0)),
        "tune_hits": int(rule.get("tune_hits", 0)),
        "tune_selected": int(rule.get("tune_selected", 0)),
    }


def build_no_position_7_prediction(
    clean_df: pd.DataFrame,
    model_scores: dict[str, np.ndarray],
    ensemble_scores: np.ndarray,
    data_report: dict[str, Any] | None = None,
    generated_at: str | None = None,
    score_weights: dict[str, float] | None = None,
) -> dict[str, Any]:
    core = get_core()
    hist = clean_df if "bai" in clean_df.columns else core.add_number_columns(clean_df)
    confidence_rule = get_active_no_position_7_rule()
    weights = score_weights or dict(confidence_rule.get("weights", NO_POSITION_7_SCORE_WEIGHTS))
    source_model = str(confidence_rule.get("model", ""))
    source_scores = active_score_items(model_scores or {}).get(source_model, ensemble_scores)
    components = build_digit_score_components(source_scores, hist)
    ranking = build_danma_ranking(model_scores, components, include_sources=True, score_weights=weights)
    digits = [int(value) for value in ranking.head(7)["digit"].tolist()]
    data_report = data_report or {}
    latest_issue = str(hist.iloc[-1].get("issue", "")) if not hist.empty else ""
    latest_date = str(data_report.get("date_max") or (hist.iloc[-1].get("date", "") if not hist.empty else ""))
    return {
        "digits": digits,
        "ranking": ranking,
        "generated_at": generated_at or now_str(),
        "data_version": str(data_report.get("data_version", "")),
        "latest_issue": latest_issue,
        "latest_date": latest_date,
        "score_note": DANMA_SCORE_NOTE,
        "weights": dict(weights),
        "target_23_rate": NO_POSITION_7_TARGET_23_RATE,
        "confidence_threshold": build_confidence_threshold_status(ranking, confidence_rule, pool_size=7),
    }


def build_reason_tags(clean_df: pd.DataFrame, code: str) -> str:
    core = get_core()
    digits = [int(code[0]), int(code[1]), int(code[2])]
    hist = clean_df.copy()

    pos_means = [
        float(hist["bai"].value_counts(normalize=True).get(digits[0], 0.0)),
        float(hist["shi"].value_counts(normalize=True).get(digits[1], 0.0)),
        float(hist["ge"].value_counts(normalize=True).get(digits[2], 0.0)),
    ]
    sum_value = sum(digits)
    span_value = max(digits) - min(digits)
    type_name = core.code_type_from_digits(digits)

    sum_probs = hist["sum3"].value_counts(normalize=True)
    span_probs = hist["span"].value_counts(normalize=True)
    type_probs = hist["type"].value_counts(normalize=True)

    reasons: list[str] = []
    overall_digit_median = float(hist[["bai", "shi", "ge"]].stack().value_counts(normalize=True).median())
    if np.mean(pos_means) >= max(0.09, overall_digit_median):
        reasons.append("位置频率偏高")
    if len(sum_probs) > 2 and float(sum_probs.get(sum_value, 0.0)) >= float(sum_probs.quantile(0.70)):
        reasons.append("和值落在常见区间")
    if len(span_probs) > 2 and float(span_probs.get(span_value, 0.0)) >= float(span_probs.quantile(0.70)):
        reasons.append("跨度处于常见区间")
    if len(type_probs) > 2 and float(type_probs.get(type_name, 0.0)) >= float(type_probs.quantile(0.60)):
        reasons.append(f"结构形态为{type_name}")

    omission_notes: list[str] = []
    for pos_name, digit in zip(["bai", "shi", "ge"], digits):
        series = hist[pos_name].astype(int).to_numpy()
        last_seen = None
        for idx, value in enumerate(series):
            if int(value) == digit:
                last_seen = idx
        if last_seen is None:
            omission_notes.append(f"{pos_name}位{digit}未出现")
        else:
            gap = len(series) - 1 - last_seen
            if gap >= max(8, int(np.median(np.arange(1, gap + 1))) if gap > 0 else 0):
                omission_notes.append(f"{pos_name}位{digit}遗漏偏长")

    if omission_notes:
        reasons.append("；".join(omission_notes[:2]))
    if not reasons:
        reasons.append("由位置频率、和值分布与集成排序共同支持")
    return "；".join(reasons[:3])


def selected_windows() -> tuple[int, ...]:
    window_choices = st.session_state.get("window_choices", DEFAULT_WINDOWS)
    return tuple(sorted({int(x) for x in window_choices}))


def selected_model_flags() -> dict[str, bool]:
    stat_flags = {
        "random_baseline": st.session_state.get("use_random_baseline", True),
        "history_frequency": st.session_state.get("use_history_frequency", True),
        "bayes_smooth_frequency": st.session_state.get("use_bayes_smooth_frequency", True),
        "omission": st.session_state.get("use_omission", True),
        "sum_distribution": st.session_state.get("use_sum_distribution", True),
        "position_frequency": st.session_state.get("use_position_frequency", True),
    }
    ml_flags = {
        "logistic_regression": st.session_state.get("use_logistic_regression", True),
        "random_forest": st.session_state.get("use_random_forest", True),
        "xgboost": st.session_state.get("use_xgboost", False),
        "lightgbm": st.session_state.get("use_lightgbm", False),
        "catboost": st.session_state.get("use_catboost", False),
    }
    deep_flags = {"tabresnet": st.session_state.get("use_tabresnet", False)}
    flags = {}
    flags.update(stat_flags)
    flags.update(ml_flags)
    flags.update(deep_flags)
    return flags


def build_ml_factories(seed: int) -> dict[str, Callable[[], Any]]:
    core = get_core()
    factories: dict[str, Callable[[], Any]] = {}
    if st.session_state.get("use_logistic_regression", True):
        factories["logistic_regression"] = lambda: core.Pipeline(
            [
                ("scaler", core.StandardScaler()),
                (
                    "clf",
                    core.LogisticRegression(
                        max_iter=1000,
                        solver="lbfgs",
                        C=0.8,
                    ),
                ),
            ]
        )
    if st.session_state.get("use_random_forest", True):
        factories["random_forest"] = lambda: core.RandomForestClassifier(
            n_estimators=160,
            max_depth=8,
            min_samples_leaf=2,
            random_state=seed,
            n_jobs=-1,
        )

    boosting_flags = {
        "xgboost": st.session_state.get("use_xgboost", False),
        "lightgbm": st.session_state.get("use_lightgbm", False),
        "catboost": st.session_state.get("use_catboost", False),
    }
    boost_factories = core.optional_boosting_factories(seed)
    for name, enabled in boosting_flags.items():
        if enabled and name in boost_factories:
            factories[name] = boost_factories[name]
    return factories


def fit_selected_models(
    train_df: pd.DataFrame,
    feature_cols: list[str],
    seed: int,
    fit_deep: bool,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    core = get_core()
    x_train = train_df[feature_cols].to_numpy(dtype=float)
    y_train = train_df[["target_bai", "target_shi", "target_ge"]].to_numpy(dtype=int)
    model_scores: dict[str, np.ndarray] = {}
    model_bundle: dict[str, Any] = {}

    ml_factories = build_ml_factories(seed)
    for name, factory in ml_factories.items():
        bundle = core.fit_digit_bundle(name, factory, x_train, y_train)
        if bundle is not None:
            model_bundle[name] = bundle

    if should_fit_deep_model(len(train_df), fit_deep):
        torch_config = core.TorchConfig(
            enabled=True,
            epochs=int(st.session_state.get("torch_epochs", 12)),
            batch_size=int(st.session_state.get("torch_batch_size", 32)),
            learning_rate=float(st.session_state.get("torch_learning_rate", 1e-3)),
            hidden_dim=int(st.session_state.get("torch_hidden_dim", 128)),
            dropout=float(st.session_state.get("torch_dropout", 0.15)),
            weight_decay=float(st.session_state.get("torch_weight_decay", 1e-4)),
        )
        torch_probs = core.fit_torch_tabresnet(x_train, y_train, x_train, torch_config, seed=seed)
        if torch_probs is not None:
            row_probs = (torch_probs[0][0], torch_probs[1][0], torch_probs[2][0])
            model_scores["tabresnet"] = core.code_scores_from_position_probs(row_probs)

    model_scores["__model_bundle__"] = model_bundle
    return model_scores, model_bundle


def get_stat_model_scores(history: pd.DataFrame, seed: int) -> dict[str, np.ndarray]:
    core = get_core()
    all_scores = core.score_statistical_models(history, seed=seed)
    flags = selected_model_flags()
    selected = {name: scores for name, scores in all_scores.items() if flags.get(name, True)}
    selected.setdefault("random_baseline", all_scores["random_baseline"])
    return selected


def train_full_model_scores(
    clean_df: pd.DataFrame,
    feature_df: pd.DataFrame,
    seed: int,
    fit_deep: bool,
) -> dict[str, np.ndarray]:
    core = get_core()
    feature_cols = [col for col in feature_df.columns if col not in {"row_index", "target_number", "target_bai", "target_shi", "target_ge"}]
    x_train = feature_df[feature_cols].to_numpy(dtype=float)
    y_train = feature_df[["target_bai", "target_shi", "target_ge"]].to_numpy(dtype=int)
    next_row = pd.DataFrame([core.build_history_feature(clean_df, windows=selected_windows())])
    next_row = next_row.reindex(columns=feature_cols, fill_value=0.0)
    x_next = next_row.to_numpy(dtype=float)

    model_scores = get_stat_model_scores(clean_df, seed=seed)

    ml_factories = build_ml_factories(seed)
    for name, factory in ml_factories.items():
        bundle = core.fit_digit_bundle(name, factory, x_train, y_train)
        if bundle is None:
            continue
        pos_probs = bundle.predict_position_probs(x_next)
        row_probs = (pos_probs[0][0], pos_probs[1][0], pos_probs[2][0])
        model_scores[name] = core.code_scores_from_position_probs(row_probs)

    if should_fit_deep_model(len(feature_df), fit_deep):
        torch_config = core.TorchConfig(
            enabled=True,
            epochs=int(st.session_state.get("torch_epochs", 12)),
            batch_size=int(st.session_state.get("torch_batch_size", 32)),
            learning_rate=float(st.session_state.get("torch_learning_rate", 1e-3)),
            hidden_dim=int(st.session_state.get("torch_hidden_dim", 128)),
            dropout=float(st.session_state.get("torch_dropout", 0.15)),
            weight_decay=float(st.session_state.get("torch_weight_decay", 1e-4)),
        )
        torch_probs = core.fit_torch_tabresnet(x_train, y_train, x_next, torch_config, seed=seed)
        if torch_probs is not None:
            row_probs = (torch_probs[0][0], torch_probs[1][0], torch_probs[2][0])
            model_scores["tabresnet"] = core.code_scores_from_position_probs(row_probs)

    return model_scores


def position_7_confidence_value(position_report: dict[str, Any], feature: str) -> float:
    positions = position_report.get("positions", {})
    top7_shares: list[float] = []
    bottom3_shares: list[float] = []
    gaps: list[float] = []
    spreads: list[float] = []
    for key in ["bai", "shi", "ge"]:
        item = positions.get(key, {})
        scores = item.get("scores", {})
        values = np.array([float(scores.get(digit, 0.0)) for digit in range(10)], dtype=float)
        total = float(values.sum())
        probs = values / total if total > 0 else np.ones(10, dtype=float) / 10.0
        order = np.argsort(probs)[::-1]
        top = order[:7]
        bottom = order[7:]
        top7_shares.append(float(probs[top].sum()))
        bottom3_shares.append(float(probs[bottom].sum()))
        gaps.append(float(probs[order[6]] - probs[order[7]]))
        spreads.append(float(probs[top].mean() - probs[bottom].mean()))
    feature_values = {
        "min_top7_share": min(top7_shares) if top7_shares else 0.0,
        "mean_top7_share": float(np.mean(top7_shares)) if top7_shares else 0.0,
        "max_bottom3_share": max(bottom3_shares) if bottom3_shares else 0.0,
        "mean_bottom3_share": float(np.mean(bottom3_shares)) if bottom3_shares else 0.0,
        "min_gap": min(gaps) if gaps else 0.0,
        "mean_gap": float(np.mean(gaps)) if gaps else 0.0,
        "min_spread": min(spreads) if spreads else 0.0,
        "mean_spread": float(np.mean(spreads)) if spreads else 0.0,
    }
    return float(feature_values.get(feature, 0.0))


def build_position_7_threshold_status(position_report: dict[str, Any], rule: dict[str, Any]) -> dict[str, Any]:
    feature = str(rule.get("feature", ""))
    direction = str(rule.get("direction", ""))
    threshold = rule.get("threshold")
    value = position_7_confidence_value(position_report, feature)
    if threshold is None or direction == "all":
        passes = True
    elif direction == "<=":
        passes = value <= float(threshold)
    elif direction == ">=":
        passes = value >= float(threshold)
    else:
        passes = False
    return {
        "model": str(rule.get("model", "")),
        "feature": feature,
        "direction": direction,
        "threshold": threshold,
        "value": float(value),
        "passes": bool(passes),
        "target_rate": float(rule.get("target_rate", 0.0)),
        "validation_rate": float(rule.get("validation_rate", 0.0)),
        "validation_hits": int(rule.get("validation_hits", 0)),
        "validation_selected": int(rule.get("validation_selected", 0)),
        "validation_samples": int(rule.get("validation_samples", 0)),
        "validation_metric": str(rule.get("validation_metric", "")),
        "at_least_2_rate": float(rule.get("at_least_2_rate", 0.0)),
        "at_least_2_hits": int(rule.get("at_least_2_hits", 0)),
        "tune_rate": float(rule.get("tune_rate", 0.0)),
        "tune_hits": int(rule.get("tune_hits", 0)),
        "tune_selected": int(rule.get("tune_selected", 0)),
        "tune_samples": int(rule.get("tune_samples", 0)),
    }


def build_position_7_code_report(ensemble_scores: np.ndarray, confidence_rule: dict[str, Any] | None = None) -> dict[str, Any]:
    core = get_core()
    marginals = core.position_marginals_from_code_scores(ensemble_scores)
    positions = [
        ("bai", "百位"),
        ("shi", "十位"),
        ("ge", "个位"),
    ]
    report: dict[str, Any] = {
        "pool_size": 7 * 7 * 7,
        "random_coverage": THEORETICAL_POSITION_TOP7_ALL_BASELINE,
        "positions": {},
    }
    for idx, (key, label) in enumerate(positions):
        scores = np.asarray(marginals[idx], dtype=float)
        order = np.argsort(scores)[::-1]
        top7 = [int(digit) for digit in order[:7]]
        excluded = [int(digit) for digit in order[7:]]
        report["positions"][key] = {
            "label": label,
            "digits": top7,
            "excluded": excluded,
            "scores": {int(digit): float(scores[int(digit)]) for digit in range(10)},
        }
    if confidence_rule:
        report["confidence_threshold"] = build_position_7_threshold_status(report, confidence_rule)
    return report


def build_position_pool_dataframe(
    clean_df: pd.DataFrame,
    model_scores: dict[str, np.ndarray],
    ensemble_scores: np.ndarray,
    position_report: dict[str, Any],
) -> pd.DataFrame:
    core = get_core()
    positions = position_report.get("positions", {})
    bai_digits = set(positions.get("bai", {}).get("digits", []))
    shi_digits = set(positions.get("shi", {}).get("digits", []))
    ge_digits = set(positions.get("ge", {}).get("digits", []))
    rows: list[dict[str, Any]] = []
    usable_scores = {name: core.calibrate_model_scores(scores) for name, scores in active_score_items(model_scores).items()}
    for idx, code in enumerate(core.CODES):
        digits = [int(code[0]), int(code[1]), int(code[2])]
        if digits[0] not in bai_digits or digits[1] not in shi_digits or digits[2] not in ge_digits:
            continue
        desc = core.describe_code(str(code))
        support = sorted(
            [(name, float(scores[idx])) for name, scores in usable_scores.items()],
            key=lambda x: x[1],
            reverse=True,
        )
        top_support = support[:3]
        rows.append(
            {
                "number": str(code),
                "score": float(ensemble_scores[idx]),
                "sum": int(desc["sum"]),
                "span": int(desc["span"]),
                "type": desc["type"],
                "source_models": " / ".join(display_model_name(name) for name, _value in top_support),
                "reason": build_reason_tags(clean_df, str(code)),
                "support": {name: float(scores[idx]) for name, scores in usable_scores.items()},
            }
        )
    pool_df = pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)
    if not pool_df.empty:
        pool_df.insert(0, "rank", np.arange(1, len(pool_df) + 1))
    return pool_df


def predict_full_scores(
    clean_df: pd.DataFrame,
    feature_df: pd.DataFrame,
    seed: int,
    fit_deep: bool,
    backtest_weights: dict[str, float] | None = None,
    model_scores: dict[str, np.ndarray] | None = None,
) -> tuple[dict[str, np.ndarray], np.ndarray, pd.DataFrame, dict[str, Any], pd.DataFrame, dict[str, pd.DataFrame]]:
    core = get_core()
    if model_scores is None:
        model_scores = train_full_model_scores(
            clean_df=clean_df,
            feature_df=feature_df,
            seed=seed,
            fit_deep=fit_deep,
        )

    ensemble = core.ensemble_scores(model_scores, weights=backtest_weights)
    position_rule = get_active_position_7_rule()
    position_weights = position_rule.get("weights")
    if position_weights:
        position_source_scores = core.ensemble_scores(active_score_items(model_scores or {}), weights=position_weights)
    else:
        position_model = str(position_rule.get("model", ""))
        position_source_scores = active_score_items(model_scores or {}).get(position_model, ensemble)
    position_report = build_position_7_code_report(position_source_scores, position_rule)
    pool_df = build_position_pool_dataframe(clean_df, model_scores, ensemble, position_report)
    top_k = int(st.session_state.get("top_k", 20))
    candidate_groups = build_candidate_groups(pool_df, top_k=top_k)
    candidate_df = candidate_groups["main"].copy()
    return model_scores, ensemble, candidate_df, position_report, pool_df, candidate_groups


def build_candidate_dataframe(
    clean_df: pd.DataFrame,
    model_scores: dict[str, np.ndarray],
    ensemble_scores: np.ndarray,
    top_k: int,
    pool_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    core = get_core()
    if pool_df is not None and not pool_df.empty:
        return pool_df.head(top_k).copy().reset_index(drop=True)

    order = np.argsort(ensemble_scores)[::-1][:top_k]
    rows: list[dict[str, Any]] = []
    usable_scores = active_score_items(model_scores)
    for rank, idx in enumerate(order, start=1):
        code = str(core.CODES[idx])
        desc = core.describe_code(code)
        support = sorted(
            [(name, float(scores[idx])) for name, scores in usable_scores.items()],
            key=lambda x: x[1],
            reverse=True,
        )
        top_support = support[:3]
        rows.append(
            {
                "rank": rank,
                "number": code,
                "score": float(ensemble_scores[idx]),
                "sum": int(desc["sum"]),
                "span": int(desc["span"]),
                "type": desc["type"],
                "source_models": " / ".join(display_model_name(name) for name, _value in top_support),
                "reason": build_reason_tags(clean_df, code),
                "support": {name: float(scores[idx]) for name, scores in usable_scores.items()},
            }
        )
    return pd.DataFrame(rows)


def code_distance(left: str, right: str) -> int:
    return sum(1 for a, b in zip(str(left), str(right)) if a != b)


def pick_diverse_candidates(pool_df: pd.DataFrame, top_k: int, min_distance: int = 2) -> pd.DataFrame:
    if pool_df.empty:
        return pool_df.copy()
    selected: list[pd.Series] = []
    used_numbers: list[str] = []
    for _idx, row in pool_df.iterrows():
        number = str(row["number"])
        if all(code_distance(number, used) >= min_distance for used in used_numbers):
            selected.append(row)
            used_numbers.append(number)
        if len(selected) >= top_k:
            break
    if len(selected) < top_k:
        selected_numbers = set(used_numbers)
        for _idx, row in pool_df.iterrows():
            number = str(row["number"])
            if number in selected_numbers:
                continue
            selected.append(row)
            selected_numbers.add(number)
            if len(selected) >= top_k:
                break
    result = pd.DataFrame(selected).copy().reset_index(drop=True)
    if not result.empty:
        result["rank"] = np.arange(1, len(result) + 1)
    return result


def pick_long_tail_candidates(pool_df: pd.DataFrame, top_k: int) -> pd.DataFrame:
    if pool_df.empty:
        return pool_df.copy()
    start = min(max(top_k * 2, len(pool_df) // 2), max(len(pool_df) - top_k, 0))
    tail_df = pool_df.iloc[start:].copy()
    if len(tail_df) < top_k:
        tail_df = pool_df.tail(top_k).copy()
    result = pick_diverse_candidates(tail_df, top_k=top_k, min_distance=2)
    if not result.empty:
        result["rank"] = np.arange(1, len(result) + 1)
    return result


def build_candidate_groups(pool_df: pd.DataFrame, top_k: int) -> dict[str, pd.DataFrame]:
    if pool_df is None or pool_df.empty:
        empty = pd.DataFrame()
        return {"main": empty, "diverse": empty, "long_tail": empty}
    main_df = pool_df.head(top_k).copy().reset_index(drop=True)
    if not main_df.empty:
        main_df["rank"] = np.arange(1, len(main_df) + 1)
    diverse_df = pick_diverse_candidates(pool_df, top_k=top_k, min_distance=2)
    long_tail_df = pick_long_tail_candidates(pool_df, top_k=top_k)
    return {"main": main_df, "diverse": diverse_df, "long_tail": long_tail_df}


def evaluate_fold_metrics(actual_numbers: list[str], score_rows: list[np.ndarray]) -> dict[str, float]:
    core = get_core()
    return core.evaluate_score_rows(actual_numbers, score_rows)


def evaluate_danma_predictions(actual_numbers: list[str], danma_rows: list[list[int]]) -> dict[str, float]:
    samples = min(len(actual_numbers), len(danma_rows))
    if samples <= 0:
        return {
            "danma_samples": 0,
            "danma_at_least_1_position_hit_rate": 0.0,
            "danma_at_least_2_position_hit_rate": 0.0,
            "danma_all_3_positions_in_set_rate": 0.0,
            "danma_all_3_danma_in_unique_draw_rate": 0.0,
            "danma_average_position_hit_count": 0.0,
        }

    at_least_1 = 0
    at_least_2 = 0
    all_3_positions = 0
    all_3_danma_unique = 0
    total_position_hits = 0
    for actual, danma in zip(actual_numbers[:samples], danma_rows[:samples]):
        actual_digits = [int(char) for char in str(actual).zfill(3)[:3]]
        danma_set = {int(digit) for digit in danma[:3]}
        position_hits = sum(1 for digit in actual_digits if digit in danma_set)
        total_position_hits += position_hits
        if position_hits >= 1:
            at_least_1 += 1
        if position_hits >= 2:
            at_least_2 += 1
        if position_hits == 3:
            all_3_positions += 1
        if len(danma_set) == 3 and danma_set.issubset(set(actual_digits)):
            all_3_danma_unique += 1

    return {
        "danma_samples": samples,
        "danma_at_least_1_position_hit_rate": at_least_1 / samples,
        "danma_at_least_2_position_hit_rate": at_least_2 / samples,
        "danma_all_3_positions_in_set_rate": all_3_positions / samples,
        "danma_all_3_danma_in_unique_draw_rate": all_3_danma_unique / samples,
        "danma_average_position_hit_count": total_position_hits / samples,
    }


def evaluate_no_position_7_predictions(actual_numbers: list[str], digit_rows: list[list[int]]) -> dict[str, float]:
    samples = min(len(actual_numbers), len(digit_rows))
    if samples <= 0:
        return {
            "no_position_7_samples": 0,
            "no_position_7_at_least_1_position_hit_rate": 0.0,
            "no_position_7_at_least_2_position_hit_rate": 0.0,
            "no_position_7_all_3_positions_in_set_rate": 0.0,
            "no_position_7_average_position_hit_count": 0.0,
        }

    at_least_1 = 0
    at_least_2 = 0
    all_3_positions = 0
    total_position_hits = 0
    for actual, digits in zip(actual_numbers[:samples], digit_rows[:samples]):
        actual_digits = [int(char) for char in str(actual).zfill(3)[:3]]
        digit_set = {int(digit) for digit in digits[:7]}
        position_hits = sum(1 for digit in actual_digits if digit in digit_set)
        total_position_hits += position_hits
        if position_hits >= 1:
            at_least_1 += 1
        if position_hits >= 2:
            at_least_2 += 1
        if position_hits == 3:
            all_3_positions += 1

    return {
        "no_position_7_samples": samples,
        "no_position_7_at_least_1_position_hit_rate": at_least_1 / samples,
        "no_position_7_at_least_2_position_hit_rate": at_least_2 / samples,
        "no_position_7_all_3_positions_in_set_rate": all_3_positions / samples,
        "no_position_7_average_position_hit_count": total_position_hits / samples,
    }


def build_danma_rows_for_backtest(
    clean_df: pd.DataFrame,
    original_indices: list[int],
    score_rows: list[np.ndarray],
    model_name: str,
) -> list[list[int]]:
    rows: list[list[int]] = []
    history_component_cache: dict[int, dict[str, np.ndarray]] = {}
    for original_idx, scores in zip(original_indices, score_rows):
        original_idx = int(original_idx)
        if original_idx not in history_component_cache:
            history = clean_df.iloc[:original_idx]
            history_component_cache[original_idx] = build_danma_history_components(history)
        position_marginal, unique_presence = digit_code_score_components(scores)
        components: dict[str, np.ndarray] = {
            "position_marginal_score": position_marginal,
            "unique_presence_score": unique_presence,
            "combination_marginal_score": normalize_digit_component((position_marginal + unique_presence) / 2.0),
        }
        components.update(history_component_cache[original_idx])
        ranking = build_danma_ranking({model_name: scores}, components, include_sources=False)
        rows.append([int(value) for value in ranking.head(3)["digit"].tolist()])
    return rows


def build_no_position_7_rows_for_backtest(
    clean_df: pd.DataFrame,
    original_indices: list[int],
    score_rows: list[np.ndarray],
    model_name: str,
    score_weights: dict[str, float] | None = None,
) -> list[list[int]]:
    rows: list[list[int]] = []
    weights = score_weights or NO_POSITION_7_SCORE_WEIGHTS
    history_component_cache: dict[int, dict[str, np.ndarray]] = {}
    for original_idx, scores in zip(original_indices, score_rows):
        original_idx = int(original_idx)
        if original_idx not in history_component_cache:
            history_component_cache[original_idx] = build_danma_history_components(clean_df.iloc[:original_idx])
        position_marginal, unique_presence = digit_code_score_components(scores)
        components: dict[str, np.ndarray] = {
            "position_marginal_score": position_marginal,
            "unique_presence_score": unique_presence,
            "combination_marginal_score": normalize_digit_component((position_marginal + unique_presence) / 2.0),
        }
        components.update(history_component_cache[original_idx])
        ranking = build_danma_ranking({model_name: scores}, components, include_sources=False, score_weights=weights)
        rows.append([int(value) for value in ranking.head(7)["digit"].tolist()])
    return rows


def merge_danma_metrics(
    base_metrics: dict[str, float],
    clean_df: pd.DataFrame,
    actual_numbers: list[str],
    original_indices: list[int],
    score_rows: list[np.ndarray],
    model_name: str,
) -> dict[str, float]:
    merged = dict(base_metrics)
    danma_rows = build_danma_rows_for_backtest(clean_df, original_indices, score_rows, model_name)
    no_position_7_rows = build_no_position_7_rows_for_backtest(clean_df, original_indices, score_rows, model_name)
    merged.update(evaluate_danma_predictions(actual_numbers, danma_rows))
    merged.update(evaluate_no_position_7_predictions(actual_numbers, no_position_7_rows))
    return merged


def run_fixed_backtest(
    clean_df: pd.DataFrame,
    feature_df: pd.DataFrame,
    seed: int,
    train_ratio: float,
    fit_deep: bool,
) -> tuple[dict[str, dict[str, float]], dict[str, float], str]:
    core = get_core()
    feature_cols = [col for col in feature_df.columns if col not in {"row_index", "target_number", "target_bai", "target_shi", "target_ge"}]
    split_at = max(5, int(len(feature_df) * train_ratio))
    split_at = min(split_at, len(feature_df) - 1)
    train_df = feature_df.iloc[:split_at].copy()
    test_df = feature_df.iloc[split_at:].copy()
    actual_numbers = test_df["target_number"].astype(str).tolist()
    original_indices = test_df["row_index"].astype(int).tolist()

    metrics: dict[str, dict[str, float]] = {}
    score_rows_by_model: dict[str, list[np.ndarray]] = {}

    # 统计模型按时间递推
    for row_number, original_idx in enumerate(original_indices):
        history = clean_df.iloc[:original_idx]
        stat_scores = get_stat_model_scores(history, seed=seed + row_number)
        for name, scores in stat_scores.items():
            score_rows_by_model.setdefault(name, []).append(scores)

    # 机器学习模型
    x_train = train_df[feature_cols].to_numpy(dtype=float)
    y_train = train_df[["target_bai", "target_shi", "target_ge"]].to_numpy(dtype=int)
    x_test = test_df[feature_cols].to_numpy(dtype=float)
    ml_factories = build_ml_factories(seed)
    for name, factory in ml_factories.items():
        bundle = core.fit_digit_bundle(name, factory, x_train, y_train)
        if bundle is None:
            continue
        pos_probs = bundle.predict_position_probs(x_test)
        for row_idx in range(len(x_test)):
            row_probs = (pos_probs[0][row_idx], pos_probs[1][row_idx], pos_probs[2][row_idx])
            score_rows_by_model.setdefault(name, []).append(core.code_scores_from_position_probs(row_probs))

    if should_fit_deep_model(len(train_df), fit_deep):
        torch_config = core.TorchConfig(
            enabled=True,
            epochs=int(st.session_state.get("torch_epochs", 12)),
            batch_size=int(st.session_state.get("torch_batch_size", 32)),
            learning_rate=float(st.session_state.get("torch_learning_rate", 1e-3)),
            hidden_dim=int(st.session_state.get("torch_hidden_dim", 128)),
            dropout=float(st.session_state.get("torch_dropout", 0.15)),
            weight_decay=float(st.session_state.get("torch_weight_decay", 1e-4)),
        )
        torch_probs = core.fit_torch_tabresnet(x_train, y_train, x_test, torch_config, seed=seed)
        if torch_probs is not None:
            for row_idx in range(len(x_test)):
                row_probs = (torch_probs[0][row_idx], torch_probs[1][row_idx], torch_probs[2][row_idx])
                score_rows_by_model.setdefault("tabresnet", []).append(core.code_scores_from_position_probs(row_probs))

    for name, rows in score_rows_by_model.items():
        metrics[name] = merge_danma_metrics(
            evaluate_fold_metrics(actual_numbers, rows),
            clean_df,
            actual_numbers,
            original_indices,
            rows,
            name,
        )

    ensemble_equal_rows = []
    for row_idx in range(len(actual_numbers)):
        current = {name: rows[row_idx] for name, rows in score_rows_by_model.items()}
        ensemble_equal_rows.append(core.ensemble_scores(current))
    metrics["ensemble_equal"] = merge_danma_metrics(
        evaluate_fold_metrics(actual_numbers, ensemble_equal_rows),
        clean_df,
        actual_numbers,
        original_indices,
        ensemble_equal_rows,
        "ensemble_equal",
    )

    weights = core.derive_weights_from_metrics(metrics)
    run_note = (
        f"回测模式：固定切分；前 {int(train_ratio * 100)}% 训练，后 {100 - int(train_ratio * 100)}% 测试；"
        f"训练样本数量 {len(train_df)}；测试样本数量 {len(test_df)}；滚动窗口数量 不适用。"
    )
    return metrics, weights, run_note


def run_rolling_backtest(
    clean_df: pd.DataFrame,
    feature_df: pd.DataFrame,
    seed: int,
    initial_train_size: int,
    batch_size: int,
    fit_deep: bool,
) -> tuple[dict[str, dict[str, float]], dict[str, float], str, pd.DataFrame]:
    core = get_core()
    if not SKLEARN_AVAILABLE:
        raise RuntimeError("当前环境缺少 scikit-learn，无法运行滚动回测。")

    feature_cols = [col for col in feature_df.columns if col not in {"row_index", "target_number", "target_bai", "target_shi", "target_ge"}]
    initial_train_size = int(initial_train_size)
    batch_size = max(1, int(batch_size))
    if initial_train_size < DEFAULT_ROLLING_INITIAL_TRAIN_SIZE:
        raise ValueError(f"初始训练窗口至少需要 {DEFAULT_ROLLING_INITIAL_TRAIN_SIZE} 期。")
    if len(feature_df) <= initial_train_size:
        raise ValueError(f"初始训练窗口 {initial_train_size} 期后没有可测试样本，请加载更多历史数据。")

    fold_rows: list[dict[str, Any]] = []
    score_rows_by_model_all: dict[str, list[np.ndarray]] = {}
    actual_numbers_by_model: dict[str, list[str]] = {}
    original_indices_by_model: dict[str, list[int]] = {}
    ensemble_rows_all: list[np.ndarray] = []
    ensemble_actual_numbers: list[str] = []
    ensemble_original_indices: list[int] = []

    test_starts = range(initial_train_size, len(feature_df), batch_size)
    for fold_idx, test_start in enumerate(test_starts, start=1):
        test_end = min(test_start + batch_size, len(feature_df))
        train_df = feature_df.iloc[:test_start].copy()
        test_df = feature_df.iloc[test_start:test_end].copy()
        x_train = train_df[feature_cols].to_numpy(dtype=float)
        y_train = train_df[["target_bai", "target_shi", "target_ge"]].to_numpy(dtype=int)
        x_test = test_df[feature_cols].to_numpy(dtype=float)
        actual_numbers = test_df["target_number"].astype(str).tolist()
        original_indices = test_df["row_index"].astype(int).tolist()

        score_rows_by_model: dict[str, list[np.ndarray]] = {}
        for row_number, original_idx in enumerate(original_indices):
            history = clean_df.iloc[:original_idx]
            stat_scores = get_stat_model_scores(history, seed=seed + fold_idx * 10 + row_number)
            for name, scores in stat_scores.items():
                score_rows_by_model.setdefault(name, []).append(scores)

        ml_factories = build_ml_factories(seed + fold_idx)
        for name, factory in ml_factories.items():
            bundle = core.fit_digit_bundle(name, factory, x_train, y_train)
            if bundle is None:
                continue
            pos_probs = bundle.predict_position_probs(x_test)
            for row_idx in range(len(x_test)):
                row_probs = (pos_probs[0][row_idx], pos_probs[1][row_idx], pos_probs[2][row_idx])
                score_rows_by_model.setdefault(name, []).append(core.code_scores_from_position_probs(row_probs))

        if should_fit_deep_model(len(train_df), fit_deep):
            torch_config = core.TorchConfig(
                enabled=True,
                epochs=int(st.session_state.get("torch_epochs", 12)),
                batch_size=int(st.session_state.get("torch_batch_size", 32)),
                learning_rate=float(st.session_state.get("torch_learning_rate", 1e-3)),
                hidden_dim=int(st.session_state.get("torch_hidden_dim", 128)),
                dropout=float(st.session_state.get("torch_dropout", 0.15)),
                weight_decay=float(st.session_state.get("torch_weight_decay", 1e-4)),
            )
            torch_probs = core.fit_torch_tabresnet(x_train, y_train, x_test, torch_config, seed=seed + fold_idx)
            if torch_probs is not None:
                for row_idx in range(len(x_test)):
                    row_probs = (torch_probs[0][row_idx], torch_probs[1][row_idx], torch_probs[2][row_idx])
                    score_rows_by_model.setdefault("tabresnet", []).append(core.code_scores_from_position_probs(row_probs))

        fold_metrics: dict[str, dict[str, float]] = {}
        for name, rows in score_rows_by_model.items():
            fold_metrics[name] = merge_danma_metrics(
                evaluate_fold_metrics(actual_numbers, rows),
                clean_df,
                actual_numbers,
                original_indices,
                rows,
                name,
            )
            score_rows_by_model_all.setdefault(name, []).extend(rows)
            actual_numbers_by_model.setdefault(name, []).extend(actual_numbers[: len(rows)])
            original_indices_by_model.setdefault(name, []).extend(original_indices[: len(rows)])

        ensemble_rows = []
        ensemble_fold_actual_numbers: list[str] = []
        ensemble_fold_original_indices: list[int] = []
        for row_idx in range(len(actual_numbers)):
            current = {name: rows[row_idx] for name, rows in score_rows_by_model.items() if row_idx < len(rows)}
            if current:
                ensemble_row = core.ensemble_scores(current)
                ensemble_rows.append(ensemble_row)
                ensemble_rows_all.append(ensemble_row)
                ensemble_actual_numbers.append(actual_numbers[row_idx])
                ensemble_original_indices.append(original_indices[row_idx])
                ensemble_fold_actual_numbers.append(actual_numbers[row_idx])
                ensemble_fold_original_indices.append(original_indices[row_idx])
        fold_metrics["ensemble_equal"] = merge_danma_metrics(
            evaluate_fold_metrics(ensemble_fold_actual_numbers, ensemble_rows),
            clean_df,
            ensemble_fold_actual_numbers,
            ensemble_fold_original_indices,
            ensemble_rows,
            "ensemble_equal",
        )

        train_start_row = int(train_df["row_index"].min())
        train_end_row = int(train_df["row_index"].max())
        test_start_row = int(test_df["row_index"].min())
        test_end_row = int(test_df["row_index"].max())
        fold_rows.append(
            {
                "fold": fold_idx,
                "train_samples": len(train_df),
                "test_samples": len(test_df),
                "train_start_row": train_start_row,
                "train_end_row": train_end_row,
                "test_start_row": test_start_row,
                "test_end_row": test_end_row,
                "train_range": f"{train_start_row}-{train_end_row}",
                "test_range": f"{test_start_row}-{test_end_row}",
                **{f"{name}_top10": fold_metrics[name]["top10_hit_rate"] for name in fold_metrics if name in fold_metrics},
                **{f"{name}_position_top7": fold_metrics[name]["position_top7_all_hit_rate"] for name in fold_metrics if name in fold_metrics},
            }
        )

    summary_metrics: dict[str, dict[str, float]] = {}
    for name, rows in score_rows_by_model_all.items():
        actual_for_model = actual_numbers_by_model.get(name, [])
        original_for_model = original_indices_by_model.get(name, [])
        summary_metrics[name] = merge_danma_metrics(
            evaluate_fold_metrics(actual_for_model, rows),
            clean_df,
            actual_for_model,
            original_for_model,
            rows,
            name,
        )
    summary_metrics["ensemble_equal"] = merge_danma_metrics(
        evaluate_fold_metrics(ensemble_actual_numbers, ensemble_rows_all),
        clean_df,
        ensemble_actual_numbers,
        ensemble_original_indices,
        ensemble_rows_all,
        "ensemble_equal",
    )

    weights = core.derive_weights_from_metrics(summary_metrics)
    fold_df = pd.DataFrame(fold_rows)
    run_note = (
        f"回测模式：滚动回测；初始训练窗口 {initial_train_size}；"
        f"训练样本数量 {initial_train_size}（首个窗口）；测试样本数量 {len(ensemble_actual_numbers)}；"
        f"滚动窗口数量 {len(fold_rows)}；每批预测期数 {batch_size}；已按时间顺序汇总所有回测结果。"
    )
    return summary_metrics, weights, run_note, fold_df


def metrics_table(metrics: dict[str, dict[str, float]]) -> pd.DataFrame:
    rows = []
    for name, metric in metrics.items():
        bai_top7 = metric.get("bai_top7_hit_rate", 0.0)
        shi_top7 = metric.get("shi_top7_hit_rate", 0.0)
        ge_top7 = metric.get("ge_top7_hit_rate", 0.0)
        position_top7 = metric.get("position_top7_all_hit_rate", 0.0)
        top10 = metric.get("top10_hit_rate", 0.0)
        top20 = metric.get("top20_hit_rate", 0.0)
        top50 = metric.get("top50_hit_rate", 0.0)
        if name == "random_baseline":
            bai_top7 = THEORETICAL_POSITION_TOP7_BASELINE
            shi_top7 = THEORETICAL_POSITION_TOP7_BASELINE
            ge_top7 = THEORETICAL_POSITION_TOP7_BASELINE
            position_top7 = THEORETICAL_POSITION_TOP7_ALL_BASELINE
            top10 = THEORETICAL_TOP10_BASELINE
            top20 = THEORETICAL_TOP20_BASELINE
            top50 = THEORETICAL_TOP50_BASELINE
        rows.append(
            {
                "模型": display_model_name(name),
                "样本": int(metric.get("samples", 0)),
                "百位Top7": bai_top7,
                "十位Top7": shi_top7,
                "个位Top7": ge_top7,
                "三位同入7码": position_top7,
                "Top10": top10,
                "Top20": top20,
                "Top50": top50,
            }
        )
    if not rows:
        return pd.DataFrame(columns=["模型", "样本", "百位Top7", "十位Top7", "个位Top7", "三位同入7码", "Top10", "Top20", "Top50"])
    return pd.DataFrame(rows).sort_values("模型").reset_index(drop=True)


DIRECT_BACKTEST_SPECS = [
    ("Top10", "top10_hit_rate", THEORETICAL_TOP10_BASELINE),
    ("Top20", "top20_hit_rate", THEORETICAL_TOP20_BASELINE),
    ("Top50", "top50_hit_rate", THEORETICAL_TOP50_BASELINE),
]


def direct_backtest_table(metrics: dict[str, dict[str, float]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for name, metric in metrics.items():
        if name == "random_baseline":
            continue
        samples = int(metric.get("samples", 0))
        row: dict[str, Any] = {"模型": display_model_name(name), "样本": samples}
        for label, key, baseline in DIRECT_BACKTEST_SPECS:
            rate = float(metric.get(key, 0.0))
            hits = int(round(rate * samples)) if samples else 0
            row[f"{label}命中率"] = f"{rate:.3%}"
            row[f"{label}命中次数"] = f"{hits}/{samples}"
            row[f"{label}随机基线"] = f"{baseline:.3%}"
        rows.append(row)
    if not rows:
        columns = ["模型", "样本"]
        for label, _key, _baseline in DIRECT_BACKTEST_SPECS:
            columns.extend([f"{label}命中率", f"{label}命中次数", f"{label}随机基线"])
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows).sort_values("模型").reset_index(drop=True)


def direct_metric_table(metrics: dict[str, dict[str, float]], label: str) -> pd.DataFrame:
    spec = next((item for item in DIRECT_BACKTEST_SPECS if item[0] == label), None)
    if spec is None:
        raise ValueError(f"未知直选指标：{label}")
    metric_label, key, baseline = spec
    rows: list[dict[str, Any]] = []
    for name, metric in metrics.items():
        if name == "random_baseline":
            continue
        samples = int(metric.get("samples", 0))
        rate = float(metric.get(key, 0.0))
        hits = int(round(rate * samples)) if samples else 0
        rows.append(
            {
                "模型": display_model_name(name),
                "样本": samples,
                f"{metric_label}命中率": f"{rate:.3%}",
                f"{metric_label}命中次数": f"{hits}/{samples}",
                f"{metric_label}随机基线": f"{baseline:.3%}",
            }
        )
    columns = ["模型", "样本", f"{metric_label}命中率", f"{metric_label}命中次数", f"{metric_label}随机基线"]
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows, columns=columns).sort_values("模型").reset_index(drop=True)


BACKTEST_AUDIT_SPECS = [
    ("百位Top7", "bai_top7_hit_rate", THEORETICAL_POSITION_TOP7_BASELINE),
    ("十位Top7", "shi_top7_hit_rate", THEORETICAL_POSITION_TOP7_BASELINE),
    ("个位Top7", "ge_top7_hit_rate", THEORETICAL_POSITION_TOP7_BASELINE),
    ("三位同时Top7", "position_top7_all_hit_rate", THEORETICAL_POSITION_TOP7_ALL_BASELINE),
    ("直选Top10", "top10_hit_rate", THEORETICAL_TOP10_BASELINE),
    ("直选Top20", "top20_hit_rate", THEORETICAL_TOP20_BASELINE),
    ("直选Top50", "top50_hit_rate", THEORETICAL_TOP50_BASELINE),
]


def wilson_interval(hits: int, samples: int, z: float = 1.96) -> tuple[float, float]:
    if samples <= 0:
        return 0.0, 0.0
    phat = hits / samples
    denom = 1.0 + z * z / samples
    center = (phat + z * z / (2 * samples)) / denom
    spread = z * math.sqrt((phat * (1 - phat) + z * z / (4 * samples)) / samples) / denom
    return max(0.0, center - spread), min(1.0, center + spread)


def normal_approx_binomial_p_value(hits: int, samples: int, baseline: float) -> float:
    if samples <= 0 or baseline <= 0.0 or baseline >= 1.0:
        return 1.0
    expected = samples * baseline
    std = math.sqrt(samples * baseline * (1.0 - baseline))
    if std <= 0:
        return 1.0
    z = abs((hits - expected) / std)
    return float(math.erfc(z / math.sqrt(2.0)))


def significance_note(samples: int, diff: float, p_value: float) -> str:
    if samples < 1000:
        return "小样本波动：不能解释成有效预测"
    if p_value < 0.05 and diff > 0:
        return "高于基线但仍需滚动窗口复核"
    if p_value < 0.05 and diff < 0:
        return "低于基线"
    return "与基线差异不显著"


def is_statistically_significant(samples: int, p_value: float) -> bool:
    return samples >= 1000 and p_value < 0.05


def backtest_audit_table(metrics: dict[str, dict[str, float]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for name, metric in metrics.items():
        if name == "random_baseline":
            continue
        samples = int(metric.get("samples", 0))
        for label, key, baseline in BACKTEST_AUDIT_SPECS:
            rate = float(metric.get(key, 0.0))
            hits = int(round(rate * samples)) if samples else 0
            expected = baseline * samples
            diff = hits - expected
            ci_low, ci_high = wilson_interval(hits, samples)
            p_value = normal_approx_binomial_p_value(hits, samples, baseline)
            rows.append(
                {
                    "模型": display_model_name(name),
                    "指标": label,
                    "命中率": f"{rate:.3%}",
                    "命中次数": f"{hits}/{samples}",
                    "理论期望": f"{expected:.1f}",
                    "差值": f"{diff:+.1f}",
                    "随机基线": f"{baseline:.3%}",
                    "95%置信区间": f"[{ci_low:.3%}, {ci_high:.3%}]",
                    "近似p值": f"{p_value:.3f}",
                    "是否显著": "是" if is_statistically_significant(samples, p_value) else "否",
                    "显著性提示": significance_note(samples, diff, p_value),
                }
            )
    if not rows:
        return pd.DataFrame(
            columns=["模型", "指标", "命中率", "命中次数", "理论期望", "差值", "随机基线", "95%置信区间", "近似p值", "是否显著", "显著性提示"]
        )
    return pd.DataFrame(rows).sort_values(["模型", "指标"]).reset_index(drop=True)


def danma_metric_conclusion(rate: float, baseline: float, samples: int, p_value: float) -> str:
    diff = rate - baseline
    if samples < 1000:
        return "样本不足1000期，不能证明长期有效"
    if p_value < 0.05 and diff > 0:
        return "高于随机基线且达到显著"
    if p_value < 0.05 and diff < 0:
        return "低于随机基线且达到显著"
    if diff > 0:
        return "略高于随机基线，但不显著"
    if diff < 0:
        return "未超过随机基线"
    return "接近随机基线"


def danma_backtest_audit_table(metrics: dict[str, dict[str, float]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for name, metric in metrics.items():
        if name == "random_baseline":
            continue
        samples = int(metric.get("danma_samples", metric.get("samples", 0)))
        for label, key, baseline in DANMA_BACKTEST_SPECS:
            rate = float(metric.get(key, 0.0))
            hits = int(round(rate * samples)) if samples else 0
            expected = baseline * samples
            diff_count = hits - expected
            diff_rate = rate - baseline
            ci_low, ci_high = wilson_interval(hits, samples)
            p_value = normal_approx_binomial_p_value(hits, samples, baseline)
            significant = is_statistically_significant(samples, p_value)
            rows.append(
                {
                    "模型": display_model_name(name),
                    "指标": label,
                    "实际命中率": f"{rate:.3%}",
                    "命中次数": f"{hits}/{samples}",
                    "理论期望": f"{expected:.1f}",
                    "随机基线": f"{baseline:.3%}",
                    "差值": f"{diff_rate:+.3%}",
                    "差值次数": f"{diff_count:+.1f}",
                    "95%置信区间": f"[{ci_low:.3%}, {ci_high:.3%}]",
                    "近似p值": f"{p_value:.3f}",
                    "是否显著": "是" if significant else "否",
                    "结论": danma_metric_conclusion(rate, baseline, samples, p_value),
                }
            )
        average_value = float(metric.get("danma_average_position_hit_count", 0.0))
        position_trials = samples * 3
        position_hits = int(round(average_value * samples)) if samples else 0
        average_expected = THEORETICAL_DANMA_AVERAGE_POSITION_HITS_BASELINE * samples
        position_ci_low, position_ci_high = wilson_interval(position_hits, position_trials)
        average_ci_low = position_ci_low * 3.0
        average_ci_high = position_ci_high * 3.0
        average_p_value = normal_approx_binomial_p_value(position_hits, position_trials, 3 / 10)
        average_significant = is_statistically_significant(samples, average_p_value)
        rows.append(
            {
                "模型": display_model_name(name),
                "指标": "平均位置命中数",
                "实际命中率": f"{average_value:.3f}",
                "命中次数": f"{position_hits}/{position_trials}",
                "理论期望": f"{average_expected:.1f}",
                "随机基线": f"{THEORETICAL_DANMA_AVERAGE_POSITION_HITS_BASELINE:.3f}",
                "差值": f"{average_value - THEORETICAL_DANMA_AVERAGE_POSITION_HITS_BASELINE:+.3f}",
                "差值次数": f"{position_hits - average_expected:+.1f}",
                "95%置信区间": f"[{average_ci_low:.3f}, {average_ci_high:.3f}]",
                "近似p值": f"{average_p_value:.3f}",
                "是否显著": "是" if average_significant else "否",
                "结论": danma_metric_conclusion(
                    average_value,
                    THEORETICAL_DANMA_AVERAGE_POSITION_HITS_BASELINE,
                    samples,
                    average_p_value,
                ),
            }
        )
    if not rows:
        return pd.DataFrame(
            columns=["模型", "指标", "实际命中率", "命中次数", "理论期望", "随机基线", "差值", "差值次数", "95%置信区间", "近似p值", "是否显著", "结论"]
        )
    return pd.DataFrame(rows).sort_values(["模型", "指标"]).reset_index(drop=True)


def danma_statistical_conclusion(metrics: dict[str, dict[str, float]]) -> str:
    current = current_generation_metric(metrics)
    if current is None:
        return "统计结论：未达到显著优势"
    name, metric = current
    audit_df = danma_backtest_audit_table({name: metric})
    if audit_df.empty:
        return "统计结论：未达到显著优势"
    for row in audit_df.to_dict("records"):
        if row.get("是否显著") == "是" and str(row.get("差值", "")).startswith("+"):
            return "统计结论：达到显著优势"
    return "统计结论：未达到显著优势"


def no_position_7_metric_conclusion(rate: float, baseline: float, samples: int, p_value: float, target_rate: float | None = None) -> str:
    diff = rate - baseline
    if target_rate is not None and rate < target_rate:
        return f"未达到{NO_POSITION_7_TARGET_23_RATE:.0%}目标，按回测如实展示"
    if samples < 1000:
        return "样本不足1000期，不能证明长期有效"
    if p_value < 0.05 and diff > 0:
        return "高于随机基线且达到显著"
    if p_value < 0.05 and diff < 0:
        return "低于随机基线且达到显著"
    if diff > 0:
        return "略高于随机基线，但不显著"
    if diff < 0:
        return "未超过随机基线"
    return "接近随机基线"


def no_position_7_backtest_audit_table(metrics: dict[str, dict[str, float]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for name, metric in metrics.items():
        if name == "random_baseline":
            continue
        samples = int(metric.get("no_position_7_samples", metric.get("samples", 0)))
        for label, key, baseline in NO_POSITION_7_BACKTEST_SPECS:
            rate = float(metric.get(key, 0.0))
            hits = int(round(rate * samples)) if samples else 0
            expected = baseline * samples
            diff_count = hits - expected
            diff_rate = rate - baseline
            ci_low, ci_high = wilson_interval(hits, samples)
            p_value = normal_approx_binomial_p_value(hits, samples, baseline)
            significant = is_statistically_significant(samples, p_value)
            target_rate = NO_POSITION_7_TARGET_23_RATE if key == "no_position_7_at_least_2_position_hit_rate" else None
            rows.append(
                {
                    "模型": display_model_name(name),
                    "指标": label,
                    "实际命中率": f"{rate:.3%}",
                    "命中次数": f"{hits}/{samples}",
                    "理论期望": f"{expected:.1f}",
                    "随机基线": f"{baseline:.3%}",
                    "差值": f"{diff_rate:+.3%}",
                    "差值次数": f"{diff_count:+.1f}",
                    "95%置信区间": f"[{ci_low:.3%}, {ci_high:.3%}]",
                    "近似p值": f"{p_value:.3f}",
                    "是否显著": "是" if significant else "否",
                    f"目标{NO_POSITION_7_TARGET_23_RATE:.0%}": "达到" if target_rate is not None and rate >= target_rate else ("未达到" if target_rate is not None else "不适用"),
                    "结论": no_position_7_metric_conclusion(rate, baseline, samples, p_value, target_rate),
                }
            )
        average_value = float(metric.get("no_position_7_average_position_hit_count", 0.0))
        position_trials = samples * 3
        position_hits = int(round(average_value * samples)) if samples else 0
        average_expected = THEORETICAL_NO_POSITION_7_AVERAGE_POSITION_HITS_BASELINE * samples
        position_ci_low, position_ci_high = wilson_interval(position_hits, position_trials)
        average_ci_low = position_ci_low * 3.0
        average_ci_high = position_ci_high * 3.0
        average_p_value = normal_approx_binomial_p_value(position_hits, position_trials, 7 / 10)
        average_significant = is_statistically_significant(samples, average_p_value)
        rows.append(
            {
                "模型": display_model_name(name),
                "指标": "平均命中位数",
                "实际命中率": f"{average_value:.3f}",
                "命中次数": f"{position_hits}/{position_trials}",
                "理论期望": f"{average_expected:.1f}",
                "随机基线": f"{THEORETICAL_NO_POSITION_7_AVERAGE_POSITION_HITS_BASELINE:.3f}",
                "差值": f"{average_value - THEORETICAL_NO_POSITION_7_AVERAGE_POSITION_HITS_BASELINE:+.3f}",
                "差值次数": f"{position_hits - average_expected:+.1f}",
                "95%置信区间": f"[{average_ci_low:.3f}, {average_ci_high:.3f}]",
                "近似p值": f"{average_p_value:.3f}",
                "是否显著": "是" if average_significant else "否",
                f"目标{NO_POSITION_7_TARGET_23_RATE:.0%}": "不适用",
                "结论": no_position_7_metric_conclusion(
                    average_value,
                    THEORETICAL_NO_POSITION_7_AVERAGE_POSITION_HITS_BASELINE,
                    samples,
                    average_p_value,
                ),
            }
        )
    if not rows:
        return pd.DataFrame(
            columns=["模型", "指标", "实际命中率", "命中次数", "理论期望", "随机基线", "差值", "差值次数", "95%置信区间", "近似p值", "是否显著", f"目标{NO_POSITION_7_TARGET_23_RATE:.0%}", "结论"]
        )
    return pd.DataFrame(rows).sort_values(["模型", "指标"]).reset_index(drop=True)


def no_position_7_statistical_conclusion(metrics: dict[str, dict[str, float]]) -> str:
    current = current_generation_metric(metrics)
    if current is None:
        return "统计结论：未达到显著优势"
    name, metric = current
    audit_df = no_position_7_backtest_audit_table({name: metric})
    if audit_df.empty:
        return "统计结论：未达到显著优势"
    for row in audit_df.to_dict("records"):
        if row.get("是否显著") == "是" and str(row.get("差值", "")).startswith("+"):
            return "统计结论：达到显著优势"
    return "统计结论：未达到显著优势"


def no_position_7_backtest_summary_table(metrics: dict[str, dict[str, float]]) -> pd.DataFrame:
    current = current_generation_metric(metrics)
    if current is None:
        return pd.DataFrame(
            [
                {"项目": "统计结论", "值": "未达到显著优势"},
                {"项目": "随机基线", "值": ">=1约97.3%，7码=23约78.4%，三位全包约34.3%"},
                {"项目": f"目标{NO_POSITION_7_TARGET_23_RATE:.0%}", "值": "等待回测"},
            ]
        )
    name, metric = current
    rate_23 = float(metric.get("no_position_7_at_least_2_position_hit_rate", 0.0))
    conclusion = no_position_7_statistical_conclusion({name: metric}).replace("统计结论：", "")
    target_status = "达到" if rate_23 >= NO_POSITION_7_TARGET_23_RATE else "未达到"
    return pd.DataFrame(
        [
            {"项目": "当前7码模型", "值": display_model_name(name)},
            {"项目": "统计结论", "值": conclusion},
            {"项目": "7码>=1准确率", "值": f"{float(metric.get('no_position_7_at_least_1_position_hit_rate', 0.0)):.3%}"},
            {"项目": "7码=23准确率", "值": f"{rate_23:.3%}"},
            {"项目": "7码三位全包", "值": f"{float(metric.get('no_position_7_all_3_positions_in_set_rate', 0.0)):.3%}"},
            {"项目": "平均命中位数", "值": f"{float(metric.get('no_position_7_average_position_hit_count', 0.0)):.3f}"},
            {"项目": "随机基线", "值": "97.3% / 78.4% / 34.3% / 2.100"},
            {"项目": f"目标{NO_POSITION_7_TARGET_23_RATE:.0%}", "值": target_status},
            {"项目": "结论", "值": f"7码=23达到{NO_POSITION_7_TARGET_23_RATE:.0%}目标，仍需继续滚动复核。" if target_status == "达到" else f"7码=23暂未达到{NO_POSITION_7_TARGET_23_RATE:.0%}目标，按回测如实展示。"},
        ]
    )


def danma_backtest_summary_table(metrics: dict[str, dict[str, float]]) -> pd.DataFrame:
    current = current_generation_metric(metrics)
    if current is None:
        return pd.DataFrame(
            [
                {"项目": "统计结论", "值": "未达到显著优势"},
                {"项目": "随机基线", "值": "至少命中1个位置约65.7%，至少命中2个位置约21.6%，三位都落入胆码集合约2.7%，3个胆码全部出现约0.6%"},
                {"项目": "结论", "值": DANMA_INEFFECTIVE_TEXT},
            ]
        )
    name, metric = current
    conclusion = danma_statistical_conclusion({name: metric}).replace("统计结论：", "")
    return pd.DataFrame(
        [
            {"项目": "当前胆码模型", "值": display_model_name(name)},
            {"项目": "统计结论", "值": conclusion},
            {"项目": "至少命中1个位置", "值": f"{float(metric.get('danma_at_least_1_position_hit_rate', 0.0)):.3%}"},
            {"项目": "至少命中2个位置", "值": f"{float(metric.get('danma_at_least_2_position_hit_rate', 0.0)):.3%}"},
            {"项目": "三位都落入胆码集合", "值": f"{float(metric.get('danma_all_3_positions_in_set_rate', 0.0)):.3%}"},
            {"项目": "3个胆码全部出现", "值": f"{float(metric.get('danma_all_3_danma_in_unique_draw_rate', 0.0)):.3%}"},
            {"项目": "平均位置命中数", "值": f"{float(metric.get('danma_average_position_hit_count', 0.0)):.3f}"},
            {"项目": "随机基线", "值": "65.7% / 21.6% / 2.7% / 0.6%"},
            {"项目": "结论", "值": DANMA_INEFFECTIVE_TEXT if conclusion != "达到显著优势" else "本次达到单项显著优势，仍需继续滚动复核。"},
        ]
    )


def metric_short_conclusion(rate: float, baseline: float, is_significant: bool) -> str:
    if rate > baseline:
        return "高于基线且达到统计显著" if is_significant else "略高于基线，但不显著"
    if rate < baseline:
        return "未超过基线"
    return "接近基线"


def baseline_summary_dataframe(metrics: dict[str, dict[str, float]]) -> pd.DataFrame:
    current = current_generation_metric(metrics)
    if current is None:
        return pd.DataFrame(
            [
                {"项目": "三位7码", "结论": "等待回测"},
                {"项目": "直选Top10", "结论": "等待回测"},
                {"项目": "综合判断", "结论": "暂未证明长期有效"},
                {"项目": "统计结论", "结论": "未达到显著优势"},
            ]
        )
    name, metric = current
    current_metrics = {name: metric}
    audit_df = backtest_audit_table(current_metrics)

    def significant_for(label: str) -> bool:
        if audit_df.empty:
            return False
        matched = audit_df[audit_df["指标"] == label]
        if matched.empty:
            return False
        return str(matched.iloc[0].get("是否显著", "否")) == "是"

    position_rate = float(metric.get("position_top7_all_hit_rate", 0.0))
    top10_rate = float(metric.get("top10_hit_rate", 0.0))
    position_conclusion = metric_short_conclusion(
        position_rate,
        THEORETICAL_POSITION_TOP7_ALL_BASELINE,
        significant_for("三位同时Top7"),
    )
    top10_conclusion = metric_short_conclusion(
        top10_rate,
        THEORETICAL_TOP10_BASELINE,
        significant_for("直选Top10"),
    )
    positive_significant = False
    if not audit_df.empty:
        for row in audit_df.to_dict("records"):
            if row.get("是否显著") == "是":
                diff_text = str(row.get("差值", "0"))
                if diff_text.startswith("+"):
                    positive_significant = True
                    break
    comprehensive = "存在单项显著优势，仍需滚动复核" if positive_significant else "暂未证明长期有效"
    statistical = "达到单项显著优势" if positive_significant else "未达到显著优势"
    return pd.DataFrame(
        [
            {"项目": "三位7码", "结论": position_conclusion},
            {"项目": "直选Top10", "结论": top10_conclusion},
            {"项目": "综合判断", "结论": comprehensive},
            {"项目": "统计结论", "结论": statistical},
        ]
    )


def position_metrics_table(metrics: dict[str, dict[str, float]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for name, metric in metrics.items():
        sample_count = int(metric.get("samples", 0))
        bai = float(metric.get("bai_top7_hit_rate", THEORETICAL_POSITION_TOP7_BASELINE if name == "random_baseline" else 0.0))
        shi = float(metric.get("shi_top7_hit_rate", THEORETICAL_POSITION_TOP7_BASELINE if name == "random_baseline" else 0.0))
        ge = float(metric.get("ge_top7_hit_rate", THEORETICAL_POSITION_TOP7_BASELINE if name == "random_baseline" else 0.0))
        all_hit = float(metric.get("position_top7_all_hit_rate", THEORETICAL_POSITION_TOP7_ALL_BASELINE if name == "random_baseline" else 0.0))
        if name == "random_baseline":
            bai = THEORETICAL_POSITION_TOP7_BASELINE
            shi = THEORETICAL_POSITION_TOP7_BASELINE
            ge = THEORETICAL_POSITION_TOP7_BASELINE
            all_hit = THEORETICAL_POSITION_TOP7_ALL_BASELINE
            conclusion = "理论基线"
        elif all_hit > THEORETICAL_POSITION_TOP7_ALL_BASELINE:
            conclusion = "小样本超过" if sample_count < 1000 else "超过"
        elif all_hit < THEORETICAL_POSITION_TOP7_ALL_BASELINE:
            conclusion = "未超过"
        else:
            conclusion = "接近基线"
        rows.append(
            {
                "模型": display_model_name(name),
                "百位Top7命中率": f"{bai:.1%}",
                "十位Top7命中率": f"{shi:.1%}",
                "个位Top7命中率": f"{ge:.1%}",
                "三位同时命中率": f"{all_hit:.1%}",
                "三位随机基线34.3%": f"{THEORETICAL_POSITION_TOP7_ALL_BASELINE:.1%}",
                "是否超过基线": conclusion,
                "样本数": sample_count,
            }
        )
    if not rows:
        return pd.DataFrame(columns=["模型", "百位Top7命中率", "十位Top7命中率", "个位Top7命中率", "三位同时命中率", "三位随机基线34.3%", "是否超过基线", "样本数"])
    return pd.DataFrame(rows).sort_values("模型").reset_index(drop=True)


def format_danma_ranking_dataframe(prediction: dict[str, Any] | None) -> pd.DataFrame:
    if not prediction:
        return pd.DataFrame(columns=["排名", "数字", "模型相对分数", "主要支持模型", "位置边际分数", "去重出现分数"])
    ranking = prediction.get("ranking")
    if ranking is None or ranking.empty:
        return pd.DataFrame(columns=["排名", "数字", "模型相对分数", "主要支持模型", "位置边际分数", "去重出现分数"])
    show_df = ranking.copy()
    for column in ["score", "position_marginal_score", "unique_presence_score"]:
        if column in show_df.columns:
            show_df[column] = show_df[column].map(lambda value: f"{float(value):.4f}")
    return show_df[["rank", "digit", "score", "source_models", "position_marginal_score", "unique_presence_score"]].rename(
        columns={
            "rank": "排名",
            "digit": "数字",
            "score": "模型相对分数",
            "source_models": "主要支持模型",
            "position_marginal_score": "位置边际分数",
            "unique_presence_score": "去重出现分数",
        }
    )


def current_danma_prediction() -> dict[str, Any] | None:
    return st.session_state.get("danma_prediction")


def current_no_position_7_prediction() -> dict[str, Any] | None:
    return st.session_state.get("no_position_7_prediction")


def append_danma_report_section(lines: list[str], prediction: dict[str, Any] | None, backtest_metrics: dict[str, dict[str, float]] | None) -> None:
    if not prediction and not backtest_metrics:
        return
    lines.append("## 下一期3个胆码")
    if prediction:
        digits = "、".join(str(digit) for digit in prediction.get("digits", []))
        lines.append(f"- 胆码：{digits}")
        lines.append(f"- 生成时间: {prediction.get('generated_at', '')}")
        lines.append(f"- 数据版本: {prediction.get('data_version', '')}")
        lines.append(f"- 最新开奖期号/日期: {prediction.get('latest_issue', '')} / {prediction.get('latest_date', '')}")
        lines.append(f"- {prediction.get('score_note', DANMA_SCORE_NOTE)}")
        threshold = prediction.get("confidence_threshold") or {}
        if threshold:
            status = threshold_status_text(threshold)
            lines.append(
                f"- 高置信阈值：模型 {display_model_name(str(threshold.get('model', '')))}；"
                f"{format_confidence_rule_text(threshold, include_values=True)}；状态 {status}。"
            )
            lines.append(
                f"- 阈值验证：至少命中1个 {float(threshold.get('validation_rate', 0.0)):.3%}，"
                f"{int(threshold.get('validation_hits', 0))}/{int(threshold.get('validation_selected', 0))}，"
                f"覆盖 {int(threshold.get('validation_selected', 0))}/{int(threshold.get('validation_samples', 0))} 期。"
            )
        ranking = prediction.get("ranking")
        if ranking is not None and not ranking.empty:
            lines.append("")
            lines.append("### 0-9完整分数排名")
            for row in ranking.head(10).to_dict("records"):
                lines.append(
                    f"- 第{int(row.get('rank', 0))}名 数字{int(row.get('digit', 0))} "
                    f"分数{float(row.get('score', 0.0)):.4f}，主要支持：{row.get('source_models', '')}"
                )
    else:
        lines.append("- 尚未生成下一期胆码，请先完成候选生成。")

    if backtest_metrics:
        lines.append("")
        lines.append("### 胆码回测结论")
        lines.append(f"- {danma_statistical_conclusion(backtest_metrics)}")
        summary_df = danma_backtest_summary_table(backtest_metrics)
        for row in summary_df.to_dict("records"):
            lines.append(f"- {row.get('项目')}: {row.get('值')}")
        lines.append("- 理论随机基线：3个胆码至少命中1个位置约65.7%；至少命中2个位置约21.6%；三个开奖号位置都落入胆码集合约2.7%；3个胆码全部出现约0.6%。")
        audit_df = danma_backtest_audit_table(backtest_metrics)
        if not audit_df.empty:
            lines.append("")
            lines.append("### 胆码回测统计审计")
            for row in audit_df.head(12).to_dict("records"):
                lines.append(
                    f"- {row.get('模型')} / {row.get('指标')}: 实际命中率 {row.get('实际命中率')}，"
                    f"命中次数 {row.get('命中次数')}，随机基线 {row.get('随机基线')}，"
                    f"差值 {row.get('差值')}，95%CI {row.get('95%置信区间')}，"
                    f"近似p值 {row.get('近似p值')}，是否显著 {row.get('是否显著')}，结论：{row.get('结论')}"
                )
        if danma_statistical_conclusion(backtest_metrics) != "统计结论：达到显著优势":
            lines.append(f"- {DANMA_INEFFECTIVE_TEXT}")
    lines.append(f"- {SAFETY_TEXT}")
    lines.append("")


def append_no_position_7_report_section(lines: list[str], prediction: dict[str, Any] | None, backtest_metrics: dict[str, dict[str, float]] | None) -> None:
    if not prediction and not backtest_metrics:
        return
    lines.append("## 不定位7码")
    if prediction:
        digits = "、".join(str(digit) for digit in prediction.get("digits", []))
        lines.append(f"- 当前7码：{digits}")
        lines.append(f"- 生成时间: {prediction.get('generated_at', '')}")
        lines.append(f"- 数据版本: {prediction.get('data_version', '')}")
        lines.append(f"- 最新开奖期号/日期: {prediction.get('latest_issue', '')} / {prediction.get('latest_date', '')}")
        lines.append(f"- {prediction.get('score_note', DANMA_SCORE_NOTE)}")
        ranking = prediction.get("ranking")
        if ranking is not None and not ranking.empty:
            lines.append("")
            lines.append("### 不定位7码 0-9完整排名")
            for row in ranking.head(10).to_dict("records"):
                lines.append(
                    f"- 第{int(row.get('rank', 0))}名 数字{int(row.get('digit', 0))} "
                    f"分数{float(row.get('score', 0.0)):.4f}，主要支持：{row.get('source_models', '')}"
                )
    else:
        lines.append("- 尚未生成不定位7码，请先完成候选生成。")

    if backtest_metrics:
        lines.append("")
        lines.append("### 不定位7码回测结论")
        lines.append(f"- {no_position_7_statistical_conclusion(backtest_metrics)}")
        summary_df = no_position_7_backtest_summary_table(backtest_metrics)
        for row in summary_df.to_dict("records"):
            lines.append(f"- {row.get('项目')}: {row.get('值')}")
        lines.append("- 理论随机基线：7码>=1约97.3%；7码=23约78.4%；7码三位全包约34.3%；平均命中位数约2.100。")
        audit_df = no_position_7_backtest_audit_table(backtest_metrics)
        if not audit_df.empty:
            lines.append("")
            lines.append("### 不定位7码回测统计审计")
            for row in audit_df.head(12).to_dict("records"):
                lines.append(
                    f"- {row.get('模型')} / {row.get('指标')}: 实际命中率 {row.get('实际命中率')}，"
                    f"命中次数 {row.get('命中次数')}，随机基线 {row.get('随机基线')}，"
                    f"差值 {row.get('差值')}，95%CI {row.get('95%置信区间')}，"
                    f"近似p值 {row.get('近似p值')}，是否显著 {row.get('是否显著')}，"
                    f"目标{NO_POSITION_7_TARGET_23_RATE:.0%} {row.get(f'目标{NO_POSITION_7_TARGET_23_RATE:.0%}')}，结论：{row.get('结论')}"
                )
    lines.append(f"- {SAFETY_TEXT}")
    lines.append("")


def build_report_text(
    report: dict[str, Any] | None,
    backtest_metrics: dict[str, dict[str, float]] | None,
    candidate_df: pd.DataFrame | None,
) -> str:
    lines: list[str] = []
    lines.append(f"# 福彩3D模拟分析报告")
    lines.append("")
    lines.append(f"- 生成时间: {now_str()}")
    if report:
        lines.append(f"- 数据文件: {report.get('file_name','')}")
        lines.append(f"- 数据版本: {report.get('data_version','')}")
        lines.append(f"- 完整 SHA256: {report.get('data_hash_sha256','')}")
        lines.append(f"- 原始期数: {report.get('raw_rows','')}")
        lines.append(f"- 清洗后期数: {report.get('kept_rows','')}")
        lines.append(f"- 日期范围: {report.get('date_min','')} -> {report.get('date_max','')}")
        lines.append(f"- 重复期号: {report.get('duplicate_issue_count',0)}")
        lines.append(f"- 缺失行: {report.get('missing_rows_count',0)}")
        lines.append(f"- 异常期号: {report.get('invalid_issue_count',0)}")
        lines.append(f"- 异常号码: {report.get('invalid_number_count',0)}")
        lines.append(f"- 前导0数量: {report.get('leading_zero_count',0)}")
        if report.get("warning_small_sample"):
            lines.append(f"- 样本提示: {report.get('sample_warning_text','')}")
        if int(report.get("calendar_gap_days_count", 0)) > 0:
            lines.append(f"- 日历日期缺口: {report.get('calendar_gap_days_count',0)} 天，可能包含休市日。")
    lines.append("")
    append_danma_report_section(lines, current_danma_prediction(), backtest_metrics)
    append_no_position_7_report_section(lines, current_no_position_7_prediction(), backtest_metrics)
    position_report = st.session_state.get("position_7_report")
    pool_df = st.session_state.get("candidate_pool_df")
    if position_report:
        lines.append("## 三个位7码结果")
        for key in ["bai", "shi", "ge"]:
            item = position_report.get("positions", {}).get(key, {})
            label = item.get("label", "")
            digits = "、".join(str(digit) for digit in item.get("digits", []))
            excluded = "、".join(str(digit) for digit in item.get("excluded", []))
            if label and digits:
                lines.append(f"- {label}7码: {digits}；排除参考: {excluded}")
        lines.append("- 7 × 7 × 7 = 343 组直选组合；理论随机覆盖率 = 34.3%。")
        if pool_df is not None and not pool_df.empty:
            lines.append(f"- 当前343组候选池: {len(pool_df)} 组。")
        lines.append("- 数字分数为模型相对分数，不是中奖概率。")
        lines.append("")
    candidate_groups = st.session_state.get("candidate_groups") or {}
    main_df = candidate_groups.get("main", candidate_df)
    diverse_df = candidate_groups.get("diverse", pd.DataFrame())
    long_tail_df = candidate_groups.get("long_tail", pd.DataFrame())

    def append_candidate_section(title: str, data: pd.DataFrame | None, limit: int = 20) -> None:
        if data is None or data.empty:
            return
        lines.append(title)
        for idx, row in enumerate(data.head(limit).itertuples(index=False), start=1):
            rank = getattr(row, "rank", idx)
            number = getattr(row, "number", "")
            score = float(getattr(row, "score", 0.0))
            reason = getattr(row, "reason", "")
            lines.append(f"- {rank}. {number} | 模型相对分数 {score:.6f} | {reason}")
        lines.append("")

    if backtest_metrics:
        lines.append("## 基线对比结论")
        if st.session_state.get("backtest_note"):
            lines.append(f"- 回测摘要: {st.session_state.get('backtest_note')}")
        for row in baseline_summary_dataframe(backtest_metrics).to_dict("records"):
            lines.append(f"- {row.get('项目')}: {row.get('结论')}")
        lines.append("")
        lines.append("## 位置级Top7回测")
        for name, metric in sorted(backtest_metrics.items()):
            bai_top7 = metric.get("bai_top7_hit_rate", 0.0)
            shi_top7 = metric.get("shi_top7_hit_rate", 0.0)
            ge_top7 = metric.get("ge_top7_hit_rate", 0.0)
            position_top7 = metric.get("position_top7_all_hit_rate", 0.0)
            if name == "random_baseline":
                bai_top7 = THEORETICAL_POSITION_TOP7_BASELINE
                shi_top7 = THEORETICAL_POSITION_TOP7_BASELINE
                ge_top7 = THEORETICAL_POSITION_TOP7_BASELINE
                position_top7 = THEORETICAL_POSITION_TOP7_ALL_BASELINE
            lines.append(
                f"- {display_model_name(name)}: 百位Top7 {bai_top7:.3f}, "
                f"十位Top7 {shi_top7:.3f}, "
                f"个位Top7 {ge_top7:.3f}, "
                f"三位同时 {position_top7:.3f}, "
                f"随机基线 {THEORETICAL_POSITION_TOP7_ALL_BASELINE:.3f}"
            )
        lines.append("")
        lines.append("## 回测结果")
        direct_table = direct_backtest_table(backtest_metrics)
        for row in direct_table.to_dict("records"):
            lines.append(
                f"- {row.get('模型')}: "
                f"Top10 {row.get('Top10命中率')}（{row.get('Top10命中次数')}，随机基线 {row.get('Top10随机基线')}）；"
                f"Top20 {row.get('Top20命中率')}（{row.get('Top20命中次数')}，随机基线 {row.get('Top20随机基线')}）；"
                f"Top50 {row.get('Top50命中率')}（{row.get('Top50命中次数')}，随机基线 {row.get('Top50随机基线')}）"
            )
        lines.append("")
        audit_table = backtest_audit_table(backtest_metrics)
        if not audit_table.empty:
            lines.append("## 回测统计审计")
            lines.append("- 统计审计用于检查命中次数、理论期望、差值、95%置信区间、近似p值和是否显著，避免把小样本波动解释成有效预测。")
            for row in audit_table.head(24).to_dict("records"):
                lines.append(
                    f"- {row.get('模型')} / {row.get('指标')}: "
                    f"命中 {row.get('命中次数')}，理论期望 {row.get('理论期望')}，"
                    f"差值 {row.get('差值')}，95%CI {row.get('95%置信区间')}，"
                    f"提示：{row.get('显著性提示')}"
                )
            lines.append("")
    if main_df is not None and not main_df.empty:
        lines.append("## 候选分组说明")
        lines.append("- 主候选按集成模型相对分数排序。")
        lines.append("- 分散候选尽量拉开号码结构，冷门参考来自343组候选池后段；二者只用于观察分布，不代表更可能出现。")
        lines.append("")
        append_candidate_section(f"## 主候选 Top{len(main_df)}", main_df)
        append_candidate_section(f"## 分散候选 Top{len(diverse_df)}", diverse_df)
        append_candidate_section(f"## 冷门参考 Top{len(long_tail_df)}", long_tail_df)
    if pool_df is not None and not pool_df.empty:
        lines.append("## 343组候选池")
        lines.append("- 以下为百位7码、十位7码、个位7码组合出的全部直选候选池，按模型相对分数排序。")
        for idx, row in enumerate(pool_df.itertuples(index=False), start=1):
            rank = getattr(row, "rank", idx)
            number = getattr(row, "number", "")
            score = float(getattr(row, "score", 0.0))
            reason = getattr(row, "reason", "")
            lines.append(f"- {rank}. {number} | 模型相对分数 {score:.6f} | {reason}")
    lines.append("")
    lines.append(f"## 安全提示")
    lines.append(SAFETY_TEXT)
    lines.append("所有候选结果必须标注：仅为模型模拟，不代表真实开奖结果。")
    return "\n".join(lines)


def render_topbar(report: dict[str, Any] | None) -> None:
    sync_current_mode()
    app_state: AppState = st.session_state.app_state
    has_raw_data = st.session_state.get("raw_bytes") is not None
    if report:
        period_count = str(report.get("kept_rows", 0))
        latest_date = str(report.get("date_max", "—"))
    elif has_raw_data:
        period_count = "待校验"
        latest_date = "待校验"
    else:
        period_count = "未加载"
        latest_date = "—"
    metrics = [
        ("文件", app_state.file_name or "未加载"),
        ("期数", period_count),
        ("最新日期", latest_date),
        ("数据版本", app_state.data_version or "—"),
        ("当前阶段", app_state.current_mode),
        ("最后运行", app_state.last_predict_at or app_state.last_backtest_at or app_state.last_trained_at or "—"),
    ]
    cards = "".join(
        (
            '<div class="fc3d-mini">'
            f'<div class="label">{escape(str(label))}</div>'
            f'<div class="value">{escape(str(value))}</div>'
            "</div>"
        )
        for label, value in metrics
    )
    st.markdown(f'<div class="fc3d-topbar-grid">{cards}</div>', unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="fc3d-chiprow">
          <span class="fc3d-chip"><b>免责声明</b>{SAFETY_TEXT}</span>
          <span class="fc3d-chip"><b>流程</b>上传 → 校验 → 训练 → 回测 → 候选</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_latest_summary() -> None:
    metrics = st.session_state.backtest_metrics or {}
    candidate_df = st.session_state.candidate_df
    position_report = st.session_state.get("position_7_report")
    verdict_text, verdict_class = baseline_comparison_text(metrics)
    candidate_text = "候选号码：尚未生成。"
    if candidate_df is not None and not candidate_df.empty:
        top_row = candidate_df.iloc[0]
        candidate_text = f"候选号码：已生成 {len(candidate_df)} 个，当前首位 {top_row['number']}，模型相对分数 {float(top_row['score']):.4f}。"
    st.markdown('<div class="fc3d-panel"><h3>最新结果摘要</h3>', unsafe_allow_html=True)
    st.markdown(f'<div class="{verdict_class}" style="margin-bottom:0.55rem">{verdict_text}</div>', unsafe_allow_html=True)
    if position_report:
        st.markdown(f'<div class="fc3d-note">{position_7_summary_text(position_report)}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="fc3d-note">{candidate_text} {SAFETY_TEXT}</div>', unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


def position_7_summary_text(position_report: dict[str, Any]) -> str:
    positions = position_report.get("positions", {})
    parts = []
    for key in ["bai", "shi", "ge"]:
        item = positions.get(key, {})
        label = item.get("label", "")
        digits = "、".join(str(digit) for digit in item.get("digits", []))
        if label and digits:
            parts.append(f"{label}7码：{digits}")
    if not parts:
        return "三个位7码：尚未生成。"
    threshold = position_report.get("confidence_threshold") or {}
    threshold_text = ""
    if threshold:
        status = threshold_status_text(threshold)
        threshold_text = (
            f"<br>高置信阈值：{display_model_name(str(threshold.get('model', '')))}；"
            f"{format_confidence_rule_text(threshold, include_values=True)}；状态：{status}。"
            f"近半年三位全中 {float(threshold.get('validation_rate', 0.0)):.2%}，"
            f"至少2位 {float(threshold.get('at_least_2_rate', 0.0)):.2%}，"
            f"覆盖 {int(threshold.get('validation_selected', 0))}/{int(threshold.get('validation_samples', 0))}。"
        )
    return (
        "<br>".join(parts)
        + "<br>7 × 7 × 7 = 343组，理论随机覆盖率 = 34.3%。"
        + threshold_text
        + "<br>数字下方为模型相对分数，只用于排序，不是中奖概率。"
    )


def position_7_source_html(metrics: dict[str, dict[str, float]]) -> str:
    best_position = best_metric(metrics, "position_top7_all_hit_rate") if metrics else None
    best_label = display_model_name(best_position[0]) if best_position else "待回测"
    items = [
        ("当前7码候选：集成模型", "本次页面展示"),
        (f"回测最佳7码模型：{best_label}", "历史回测对比"),
    ]
    cards = "".join(
        (
            '<div class="fc3d-mini">'
            f'<div class="label">{escape(label)}</div>'
            f'<div class="value">{escape(value)}</div>'
            '</div>'
        )
        for label, value in items
    )
    return f'<div class="fc3d-position-source">{cards}</div>'


def danma_cards_html(prediction: dict[str, Any]) -> str:
    ranking = prediction.get("ranking", pd.DataFrame())
    cards: list[str] = []
    for row in ranking.head(3).to_dict("records"):
        cards.append(
            '<div class="fc3d-danma-card">'
            f'<div class="digit">{int(row.get("digit", 0))}</div>'
            f'<div class="meta"><b>模型相对分数</b> {float(row.get("score", 0.0)):.4f}</div>'
            f'<div class="meta"><b>支持来源</b> {escape(str(row.get("source_models", "")))}</div>'
            '</div>'
        )
    return f'<div class="fc3d-danma-grid">{"".join(cards)}</div>' if cards else ""


def render_danma_module(
    prediction: dict[str, Any] | None,
    metrics: dict[str, dict[str, float]] | None = None,
    expanded_ranking: bool = True,
) -> None:
    st.markdown('<div class="fc3d-panel"><h3>下一期3个胆码</h3>', unsafe_allow_html=True)
    if not prediction:
        st.markdown(
            f'<div class="fc3d-note">生成候选后，这里会显示下一期3个胆码、0-9完整排名和胆码回测结论。{SAFETY_TEXT}</div>',
            unsafe_allow_html=True,
        )
        if metrics:
            st.markdown(f'<div class="fc3d-warning" style="margin-top:0.5rem">{danma_statistical_conclusion(metrics)}</div>', unsafe_allow_html=True)
            st.dataframe(danma_backtest_summary_table(metrics), width="stretch", hide_index=True)
        st.markdown("</div>", unsafe_allow_html=True)
        return

    digits_text = "、".join(str(digit) for digit in prediction.get("digits", []))
    generated = escape(str(prediction.get("generated_at", "")))
    data_version = escape(str(prediction.get("data_version", "")))
    latest_issue = escape(str(prediction.get("latest_issue", "")))
    latest_date = escape(str(prediction.get("latest_date", "")))
    threshold = prediction.get("confidence_threshold") or {}
    threshold_note = ""
    if threshold:
        threshold_status = threshold_status_text(threshold)
        threshold_note = (
            f"<br>当前胆码配置：{escape(display_model_name(str(threshold.get('model', ''))))}；"
            f"{escape(format_confidence_rule_text(threshold, include_values=True))}；状态：{threshold_status}。"
        )
    st.markdown(
        f"""
        <div class="fc3d-warning">胆码：<b>{escape(digits_text)}</b></div>
        <div class="fc3d-note" style="margin-top:0.45rem">
          生成时间：{generated}；数据版本：{data_version}；最新开奖期号/日期：{latest_issue} / {latest_date}。<br>
          胆码只是模型排序结果，不是中奖概率，不代表真实预测。{threshold_note}
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(danma_cards_html(prediction), unsafe_allow_html=True)
    if metrics:
        st.markdown(f'<div class="fc3d-note" style="margin-top:0.65rem"><b>{danma_statistical_conclusion(metrics)}</b></div>', unsafe_allow_html=True)
        st.dataframe(danma_backtest_summary_table(metrics), width="stretch", hide_index=True)
        audit_df = danma_backtest_audit_table(metrics)
        if not audit_df.empty:
            with st.expander("胆码回测统计审计", expanded=False):
                st.dataframe(audit_df, width="stretch", hide_index=True)
        if danma_statistical_conclusion(metrics) != "统计结论：达到显著优势":
            st.markdown(f'<div class="fc3d-warning" style="margin-top:0.5rem">{DANMA_INEFFECTIVE_TEXT}</div>', unsafe_allow_html=True)
    ranking_df = format_danma_ranking_dataframe(prediction)
    with st.expander("0-9完整排名", expanded=expanded_ranking):
        st.dataframe(ranking_df, width="stretch", hide_index=True)
    st.markdown(f'<div class="fc3d-note">{SAFETY_TEXT}</div>', unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


def render_no_position_7_module(
    prediction: dict[str, Any] | None,
    metrics: dict[str, dict[str, float]] | None = None,
    expanded_ranking: bool = False,
) -> None:
    st.markdown('<div class="fc3d-panel"><h3>不定位7码</h3>', unsafe_allow_html=True)
    if not prediction:
        st.markdown(
            f'<div class="fc3d-note">生成候选后，这里会显示不定位7码、0-9完整排名和7码=23回测结论。{SAFETY_TEXT}</div>',
            unsafe_allow_html=True,
        )
        if metrics:
            st.dataframe(no_position_7_backtest_summary_table(metrics), width="stretch", hide_index=True)
        st.markdown("</div>", unsafe_allow_html=True)
        return

    digits_text = "、".join(str(digit) for digit in prediction.get("digits", []))
    generated = escape(str(prediction.get("generated_at", "")))
    data_version = escape(str(prediction.get("data_version", "")))
    latest_issue = escape(str(prediction.get("latest_issue", "")))
    latest_date = escape(str(prediction.get("latest_date", "")))
    threshold = prediction.get("confidence_threshold") or {}
    if threshold:
        threshold_status = threshold_status_text(threshold)
        threshold_text = (
            f"高置信阈值：{escape(display_model_name(str(threshold.get('model', ''))))}；"
            f"{escape(format_confidence_rule_text(threshold, include_values=True))}；状态：{threshold_status}。"
            f"严格验证 7码=23 {float(threshold.get('validation_rate', 0.0)):.3%}，"
            f"{int(threshold.get('validation_hits', 0))}/{int(threshold.get('validation_selected', 0))}，"
            f"覆盖 {int(threshold.get('validation_selected', 0))}/{int(threshold.get('validation_samples', 0))} 期。"
        )
    else:
        threshold_text = "尚未配置高置信阈值。"
    st.markdown(
        f"""
        <div class="fc3d-warning">当前7码：<b>{escape(digits_text)}</b></div>
        <div class="fc3d-note" style="margin-top:0.45rem">
          生成时间：{generated}；数据版本：{data_version}；最新开奖期号/日期：{latest_issue} / {latest_date}。<br>
          7码=23 指命中2位或3位算成功；随机基线约78.4%，目标参考为{NO_POSITION_7_TARGET_23_RATE:.0%}。{DANMA_SCORE_NOTE}<br>
          {threshold_text}
        </div>
        """,
        unsafe_allow_html=True,
    )
    if metrics:
        st.markdown(f'<div class="fc3d-note" style="margin-top:0.65rem"><b>{no_position_7_statistical_conclusion(metrics)}</b></div>', unsafe_allow_html=True)
        st.dataframe(no_position_7_backtest_summary_table(metrics), width="stretch", hide_index=True)
        audit_df = no_position_7_backtest_audit_table(metrics)
        if not audit_df.empty:
            with st.expander("不定位7码回测统计审计", expanded=False):
                st.dataframe(audit_df, width="stretch", hide_index=True)
    ranking_df = format_danma_ranking_dataframe(prediction)
    with st.expander("不定位7码 0-9完整排名", expanded=expanded_ranking):
        st.dataframe(ranking_df, width="stretch", hide_index=True)
    st.markdown(f'<div class="fc3d-note">{SAFETY_TEXT}</div>', unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


def render_position_7_module(position_report: dict[str, Any] | None, metrics: dict[str, dict[str, float]] | None = None) -> None:
    if not position_report:
        st.markdown(
            '<div class="fc3d-panel"><h3>三个位7码结果</h3><div class="fc3d-note">生成候选后，这里会显示百位7码、十位7码、个位7码，以及 343 组理论覆盖说明。</div></div>',
            unsafe_allow_html=True,
        )
        return

    metric_text = position_7_backtest_text(metrics or {})

    positions = position_report.get("positions", {})
    cards: list[str] = []
    for key in ["bai", "shi", "ge"]:
        item = positions.get(key, {})
        label = escape(str(item.get("label", "")))
        scores = item.get("scores", {})
        digit_items = "".join(
            (
                '<span class="fc3d-digit-pill">'
                f'{int(digit)}<small>{float(scores.get(int(digit), 0.0)):.4f}</small>'
                "</span>"
            )
            for digit in item.get("digits", [])
        )
        excluded = "、".join(str(digit) for digit in item.get("excluded", []))
        recommended = "、".join(str(digit) for digit in item.get("digits", []))
        all_digit_scores = "；".join(
            f"{digit}:{float(scores.get(int(digit), 0.0)):.4f}"
            for digit in range(10)
        )
        cards.append(
            '<div class="fc3d-position-card">'
            f'<div class="title">{label}7码：{recommended}</div>'
            f'<div class="fc3d-digit-row">{digit_items}</div>'
            f'<div class="fc3d-note" style="margin-top:0.45rem">0-9相对分数：{all_digit_scores}</div>'
            f'<div class="fc3d-note" style="margin-top:0.45rem">排除参考：{excluded}。不代表一定不会开。</div>'
            '</div>'
        )
    metric_html = f'<div class="fc3d-note" style="margin-top:0.55rem">{escape(metric_text)}</div>' if metric_text else ""
    source_html = position_7_source_html(metrics or {})
    panel_html = (
        '<div class="fc3d-panel"><h3>三个位7码结果</h3>'
        '<div class="fc3d-warning">7 × 7 × 7 = 343 组直选组合；理论随机覆盖率 = 34.3%。如果只看三个位都落入7码集合，模型必须长期稳定超过这个基线，才说明有分析价值。</div>'
        f'{source_html}'
        f'{metric_html}'
        f'<div class="fc3d-position-grid">{"".join(cards)}</div>'
        '<div class="fc3d-note" style="margin-top:0.6rem">当前展示的定位7码来自快捷方案指定模型；高置信阈值只用于模拟排序和筛选参考。</div>'
        '<div class="fc3d-note" style="margin-top:0.6rem">数字下方为模型相对分数，只用于排序，不是中奖概率。</div>'
        '</div>'
    )
    st.markdown(panel_html, unsafe_allow_html=True)


def position_7_backtest_text(metrics: dict[str, dict[str, float]]) -> str:
    if not metrics:
        return "尚未完成回测，暂不能判断是否长期超过 34.3% 理论随机基线。"
    best_position = best_metric(metrics, "position_top7_all_hit_rate")
    if best_position is None:
        return "尚未得到可比较的模型回测结果。"
    best_name, metric = best_position
    best_rate = float(metric.get("position_top7_all_hit_rate", 0.0))
    sample_count = int(metric.get("samples", 0))
    hits = int(round(best_rate * sample_count)) if sample_count else 0
    expected = THEORETICAL_POSITION_TOP7_ALL_BASELINE * sample_count
    if sample_count < 1000:
        verdict = "样本较少，仅供学习参考"
    elif best_rate > THEORETICAL_POSITION_TOP7_ALL_BASELINE + 0.02:
        verdict = "本次超过理论随机基线，仍需继续观察"
    elif best_rate < THEORETICAL_POSITION_TOP7_ALL_BASELINE - 0.02:
        verdict = "本次低于理论随机基线"
    else:
        verdict = "本次与理论随机基线接近"
    return (
        f"三位7码最佳模型：{display_model_name(best_name)}，命中率 {best_rate:.3%}"
        f"（约 {hits}/{sample_count} 次，理论期望约 {expected:.1f} 次）；{verdict}。"
    )


def position_metric_summary_html(metrics: dict[str, dict[str, float]]) -> str:
    current = current_generation_metric(metrics)
    if current is None:
        return ""
    name, metric = current
    items = [
        ("当前回测口径", display_model_name(name)),
        ("百位Top7", f"{float(metric.get('bai_top7_hit_rate', 0.0)):.1%}"),
        ("十位Top7", f"{float(metric.get('shi_top7_hit_rate', 0.0)):.1%}"),
        ("个位Top7", f"{float(metric.get('ge_top7_hit_rate', 0.0)):.1%}"),
        ("三位同时", f"{float(metric.get('position_top7_all_hit_rate', 0.0)):.1%}"),
        ("随机基线", f"{THEORETICAL_POSITION_TOP7_ALL_BASELINE:.1%}"),
    ]
    cards = "".join(
        (
            '<div class="fc3d-mini">'
            f'<div class="label">{escape(label)}</div>'
            f'<div class="value">{escape(value)}</div>'
            '</div>'
        )
        for label, value in items
    )
    return f'<div class="fc3d-topbar-grid" style="margin:0.45rem 0 0.75rem 0">{cards}</div>'


def backtest_conclusion_html(metrics: dict[str, dict[str, float]], note: str, fold_df: pd.DataFrame | None = None) -> str:
    summary_df = backtest_run_summary_dataframe(metrics, note, fold_df)
    summary_map = dict(zip(summary_df.get("项目", []), summary_df.get("值", []))) if not summary_df.empty else {}
    verdict_df = baseline_summary_dataframe(metrics)

    def conclusion_for(item: str, default: str = "等待回测") -> str:
        if verdict_df.empty:
            return default
        matched = verdict_df[verdict_df["项目"] == item]
        if matched.empty:
            return default
        return str(matched.iloc[0]["结论"])

    statistic_text = conclusion_for("统计结论", "未达到显著优势")
    items = [
        ("回测模式", str(summary_map.get("回测模式", "尚未运行"))),
        ("测试样本", str(summary_map.get("测试样本数量", "—"))),
        ("滚动窗口", str(summary_map.get("滚动窗口数量", "—"))),
        ("三位7码", conclusion_for("三位7码")),
        ("直选Top10", conclusion_for("直选Top10")),
        ("统计结论", statistic_text),
    ]
    cards = "".join(
        (
            '<div class="fc3d-conclusion-card">'
            f'<div class="label">{escape(label)}</div>'
            f'<div class="value">{escape(value)}</div>'
            "</div>"
        )
        for label, value in items
    )
    warning = ""
    if statistic_text != "达到单项显著优势":
        warning = '<div class="fc3d-warning" style="margin:0.5rem 0">当前模型暂未证明长期有效，仅可作为模拟排序参考。</div>'
    return (
        '<div class="fc3d-note" style="margin-top:0.45rem"><b>回测结论卡片</b></div>'
        f'<div class="fc3d-backtest-conclusion-grid">{cards}</div>'
        f"{warning}"
    )


def significance_badges_html(audit_df: pd.DataFrame) -> str:
    if audit_df is None or audit_df.empty or "是否显著" not in audit_df.columns:
        return ""
    badges = []
    for row in audit_df.head(14).to_dict("records"):
        significant = str(row.get("是否显著", "否")) == "是"
        badge_class = "fc3d-sig-yes" if significant else "fc3d-sig-no"
        label = f"{row.get('指标', '')}：{row.get('是否显著', '否')}"
        badges.append(f'<span class="{badge_class}">{escape(str(label))}</span>')
    return f'<div class="fc3d-chiprow" style="margin:0.35rem 0 0.55rem 0">{"".join(badges)}</div>'


def compact_weight_text(weights: dict[str, float]) -> str:
    aliases = {
        "combination_marginal_score": "c",
        "near30_frequency": "n30",
        "near100_frequency": "n100",
        "position_frequency": "p",
        "omission_rebound": "o",
        "sum_distribution_support": "s",
        "heat_cold_stability": "h",
    }
    parts = []
    for key in [
        "combination_marginal_score",
        "near30_frequency",
        "near100_frequency",
        "position_frequency",
        "omission_rebound",
        "sum_distribution_support",
        "heat_cold_stability",
    ]:
        value = float(weights.get(key, 0.0))
        if abs(value) > 1e-12:
            parts.append(f"{aliases[key]}={value:g}")
    return ",".join(parts)


def direct_top20_weight_text(weights: dict[str, float]) -> str:
    ordered = [
        ("h", "history_frequency"),
        ("b", "bayes_smooth_frequency"),
        ("o", "omission"),
        ("s", "sum_distribution"),
        ("p", "position_frequency"),
    ]
    return ",".join(f"{label}={float(weights.get(key, 0.0)):g}" for label, key in ordered)


def format_threshold_condition(condition: dict[str, Any], include_value: bool = False) -> str:
    feature = str(condition.get("feature", ""))
    direction = str(condition.get("direction", ""))
    threshold = condition.get("threshold")
    if direction == "all":
        return "全量输出"
    text = f"{feature}{direction}{float(threshold):.3f}" if threshold is not None else f"{feature}{direction}"
    if include_value:
        text += f" 当前{float(condition.get('value', 0.0)):.3f}"
    return text


def format_confidence_rule_text(rule_or_status: dict[str, Any], include_values: bool = False) -> str:
    conditions = rule_or_status.get("conditions") or []
    if conditions:
        return " 且 ".join(format_threshold_condition(condition, include_value=include_values) for condition in conditions)
    return format_threshold_condition(rule_or_status, include_value=include_values)


def threshold_status_text(threshold: dict[str, Any]) -> str:
    if not threshold:
        return "未配置阈值，仅按分数从高到低排序推荐"
    if threshold.get("direction") == "all" or threshold.get("threshold") is None:
        return "全量输出，按分数从高到低排序推荐"
    return "符合阈值" if threshold.get("passes") else "未符合阈值，仅按分数从高到低排序推荐"


def quick_scheme_dataframe() -> pd.DataFrame:
    seven_rule = NO_POSITION_7_CONFIDENCE_RULE
    position_rule = POSITION_7_CONFIDENCE_RULE
    danma_rule = DANMA_CONFIDENCE_RULE
    direct_rule = DIRECT_TOP20_FULL_COVERAGE_RULE
    seven_weights = seven_rule.get("weights", NO_POSITION_7_SCORE_WEIGHTS)
    position_weights = position_rule.get("weights", POSITION_7_SCORE_WEIGHTS)
    danma_weights = danma_rule.get("weights", DANMA_SCORE_WEIGHTS)
    direct_weights = direct_rule.get("weights", DIRECT_TOP20_FULL_COVERAGE_WEIGHTS)
    return pd.DataFrame(
        [
            {
                "方案": "不定位7码高置信=23",
                "用途": "筛7码，命中2位或3位算成功",
                "权重/阈值": (
                    f"{compact_weight_text(seven_weights)}；"
                    f"{format_confidence_rule_text(seven_rule)}"
                ),
                "近一年验证": (
                    f"7码=23 {float(seven_rule.get('validation_rate', 0.0)):.2%}，"
                    f"覆盖{int(seven_rule.get('validation_selected', 0))}/{int(seven_rule.get('validation_samples', 0))}"
                ),
                "说明": "高置信阈值模式，未覆盖期不计入条件命中率",
            },
            {
                "方案": "定位7码高置信全中",
                "用途": "百十个位各7码，三位全中",
                "权重/阈值": (
                    f"{direct_top20_weight_text(position_weights)}；"
                    f"{format_confidence_rule_text(position_rule)}"
                ),
                "近一年验证": (
                    f"三位全中 {float(position_rule.get('validation_rate', 0.0)):.2%}，"
                    f"覆盖{int(position_rule.get('validation_selected', 0))}/{int(position_rule.get('validation_samples', 0))}"
                ),
                "说明": (
                    f"至少2位 {float(position_rule.get('at_least_2_rate', 0.0)):.2%}；"
                    "定位7码独立阈值"
                ),
            },
            {
                "方案": "三胆码最优独立方案",
                "用途": "筛3个胆码，至少命中1个",
                "权重/阈值": f"{compact_weight_text(danma_weights)}；{format_confidence_rule_text(danma_rule)}",
                "近一年验证": (
                    f"至少命中1个 {float(danma_rule.get('validation_rate', 0.0)):.2%}，"
                    f"覆盖{int(danma_rule.get('validation_selected', 0))}/{int(danma_rule.get('validation_samples', 0))}"
                ),
                "说明": "胆码独立权重，不共用7码阈值",
            },
            {
                "方案": "直选20注全覆盖方案",
                "用途": "直选Top20组合排序",
                "权重/阈值": f"{direct_top20_weight_text(direct_weights)}；全量输出",
                "近一年验证": (
                    f"近一年Top20 {float(direct_rule.get('validation_rate', 0.0)):.2%}，"
                    f"覆盖{int(direct_rule.get('validation_selected', 0))}/{int(direct_rule.get('validation_samples', 0))}"
                ),
                "说明": "每期输出Top20，不做高置信筛选",
            },
        ]
    )


def render_primary_actions(report: dict[str, Any] | None) -> tuple[bool, bool, bool, bool, bool, bool]:
    availability = action_availability()
    reload_required = demo_reload_required()
    confirm_reload = True
    next_label, next_hint = next_step_copy()
    next_reason = ""

    with st.container(key="main_action_panel"):
        st.markdown("<h3>快速操作</h3>", unsafe_allow_html=True)
        if report:
            candidate_df = st.session_state.get("candidate_df")
            pool_df = st.session_state.get("candidate_pool_df")
            if candidate_df is not None and not candidate_df.empty:
                candidate_note = f"候选已生成：显示 {len(candidate_df)} 个，7码候选池 {len(pool_df) if pool_df is not None else 343} 组。"
            else:
                candidate_note = "候选号码：尚未生成。"
            st.markdown(
                f'<div class="fc3d-note fc3d-nav-note">当前阶段：{app_stage()}。{candidate_note}</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div class="fc3d-note fc3d-nav-note">先{demo_button_label()}，或在侧边栏上传文件后校验。</div>',
                unsafe_allow_html=True,
            )
        if reload_required:
            st.warning("重新加载近5年开奖记录会清空当前训练、回测和候选结果。")
            confirm_reload = st.checkbox(f"确认{demo_button_label()}", key="main_confirm_reload_demo")
            if not confirm_reload:
                next_reason = "如需重新加载，请先勾选确认。"
        if not next_reason:
            unavailable_reason = disabled_action_reason(recommended_action_key())
            if unavailable_reason and not availability.get(recommended_action_key(), True):
                next_reason = unavailable_reason
        st.markdown(
            f"""
            <div class="fc3d-action-grid">
              <div class="fc3d-next-action"><span>下一步</span><b>{escape(next_label)}</b><br>{escape(next_hint)}</div>
              <div class="fc3d-disabled-reason"><span>禁用说明</span>{escape(next_reason or "灰色按钮暂不可用，请按当前阶段顺序完成。")}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        load_col, validate_col, train_col = st.columns(3, gap="small")
        backtest_col, predict_col, export_col = st.columns(3, gap="small")
        with load_col:
            demo_btn = st.button(
                demo_button_label(),
                key="main_load_demo",
                width="stretch",
                disabled=not availability["demo"] or (reload_required and not confirm_reload),
                type=action_button_type("demo", availability) if not reload_required else "secondary",
                help=disabled_action_reason("demo") if reload_required else "加载系统缓存的近5年开奖记录。",
            )
        with validate_col:
            validate_btn = st.button(
                "校验数据",
                key="main_validate_data",
                width="stretch",
                disabled=not availability["validate"],
                type=action_button_type("validate", availability),
                help=disabled_action_reason("validate") if not availability["validate"] else "正在校验数据，预计30-40秒。",
            )
        with train_col:
            train_btn = st.button(
                "训练模型",
                key="main_train_models",
                width="stretch",
                disabled=not availability["train"],
                type=action_button_type("train", availability),
                help=disabled_action_reason("train") if not availability["train"] else "正在训练模型，预计30-60秒。",
            )
        with backtest_col:
            backtest_btn = st.button(
                "开始滚动回测",
                key="main_start_backtest",
                width="stretch",
                disabled=not availability["backtest"],
                type=action_button_type("backtest", availability),
                help=disabled_action_reason("backtest") if not availability["backtest"] else "正在滚动回测，预计40-60秒。",
            )
        with predict_col:
            predict_btn = st.button(
                "生成候选与胆码",
                key="main_generate_candidates",
                width="stretch",
                disabled=not availability["predict"],
                type=action_button_type("predict", availability),
                help=disabled_action_reason("predict") if not availability["predict"] else "正在生成候选，请稍候。",
            )
        with export_col:
            report_btn = st.button(
                "导出分析报告",
                key="main_export_report",
                width="stretch",
                disabled=not availability["export"],
                type=action_button_type("export", availability),
                help=disabled_action_reason("export") if not availability["export"] else "导出完整分析报告。",
            )
        with st.expander("快捷方案", expanded=False):
            seven_col, position_col, danma_col, direct_col = st.columns(4, gap="small")
            with seven_col:
                if st.button("一键配置7码覆盖优先", key="quick_apply_no_position_7", width="stretch"):
                    st.success(apply_quick_prediction_scheme("no_position_7_coverage"))
            with position_col:
                if st.button("一键配置定位7码", key="quick_apply_position_7", width="stretch"):
                    st.success(apply_quick_prediction_scheme("position_7_high_confidence"))
            with danma_col:
                if st.button("一键配置胆码最优", key="quick_apply_danma", width="stretch"):
                    st.success(apply_quick_prediction_scheme("danma_optimal"))
            with direct_col:
                if st.button("一键配置直选20", key="quick_apply_direct_top20", width="stretch"):
                    st.success(apply_quick_prediction_scheme("direct_top20_full_coverage"))
            st.dataframe(quick_scheme_dataframe(), width="stretch", hide_index=True)
            st.markdown(f'<div class="fc3d-note">{SAFETY_TEXT}</div>', unsafe_allow_html=True)
    return demo_btn, validate_btn, train_btn, backtest_btn, predict_btn, report_btn


def render_header() -> None:
    st.markdown(f'<div class="fc3d-title">{APP_TITLE}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="fc3d-subtitle">{APP_SUBTITLE}</div>', unsafe_allow_html=True)


def render_data_cards(report: dict[str, Any]) -> None:
    st.markdown('<div class="fc3d-panel"><h3>数据概览</h3>', unsafe_allow_html=True)
    cols = st.columns(6, gap="small")
    cards = [
        ("原始期数", report.get("raw_rows", 0)),
        ("清洗后期数", report.get("kept_rows", 0)),
        ("重复期号", report.get("duplicate_issue_count", 0)),
        ("缺失行", report.get("missing_rows_count", 0)),
        ("异常号码", report.get("invalid_number_count", 0)),
        ("前导0数量", report.get("leading_zero_count", 0)),
    ]
    for col, (label, value) in zip(cols, cards):
        with col:
            st.markdown(
                f'<div class="fc3d-card"><div class="label">{label}</div><div class="value">{value}</div></div>',
                unsafe_allow_html=True,
            )
    st.markdown("</div>", unsafe_allow_html=True)


def render_validation_panel(report: dict[str, Any]) -> None:
    if not report:
        st.markdown('<div class="fc3d-panel"><h3>数据检查</h3><div class="fc3d-note">请先上传历史开奖 CSV，或点击侧边栏按钮加载近5年开奖记录，然后再校验数据。</div></div>', unsafe_allow_html=True)
        return
    st.markdown('<div class="fc3d-panel"><h3>数据检查</h3>', unsafe_allow_html=True)
    hard_issue_count = sum(
        int(report.get(key, 0))
        for key in [
            "missing_rows_count",
            "invalid_issue_count",
            "invalid_number_count",
            "invalid_date_count",
            "duplicate_issue_count",
            "date_order_issue_count",
        ]
    )
    status_class = "fc3d-good" if hard_issue_count == 0 else "fc3d-warning"
    sample_note = ""
    if report.get("warning_small_sample"):
        sample_note = f'<br><b>{escape(str(report.get("sample_warning_text", "")))}</b>'
    gap_note = ""
    if int(report.get("calendar_gap_days_count", 0)) > 0:
        gap_note = f'<br>检测到日历日期缺口 <b>{int(report.get("calendar_gap_days_count", 0))}</b> 天，可能包含休市日，已作为审计信息保留。'
    st.markdown(
        f"""
        <div class="{status_class}">
          数据版本 <b>{escape(str(report.get('data_version','')))}</b>，保留 <b>{int(report.get('kept_rows',0))}</b> 期，丢弃 <b>{int(report.get('dropped_rows',0))}</b> 行。<br>
          日期范围 <b>{escape(str(report.get('date_min','')))}</b> 到 <b>{escape(str(report.get('date_max','')))}</b>。{sample_note}{gap_note}
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div class="fc3d-note" style="margin-top:0.5rem">完整 SHA256：{escape(str(report.get("data_hash_sha256", "")))}</div>',
        unsafe_allow_html=True,
    )
    detail_cols = st.columns(2, gap="large")
    with detail_cols[0]:
        with st.expander("异常样本", expanded=False):
            st.write("缺失行")
            st.dataframe(pd.DataFrame(report.get("missing_rows", [])), width="stretch", height=180)
            st.write("异常期号")
            st.dataframe(pd.DataFrame(report.get("invalid_issue_rows", [])), width="stretch", height=180)
            st.write("异常号码")
            st.dataframe(pd.DataFrame(report.get("invalid_number_rows", [])), width="stretch", height=180)
    with detail_cols[1]:
        with st.expander("重复与日期问题", expanded=False):
            st.write("重复期号")
            st.dataframe(pd.DataFrame(report.get("duplicate_issue_rows", [])), width="stretch", height=180)
            st.write("异常日期")
            st.dataframe(pd.DataFrame(report.get("invalid_date_rows", [])), width="stretch", height=180)
            st.write("日历日期缺口")
            st.dataframe(pd.DataFrame({"缺口日期": report.get("calendar_gap_days", [])}), width="stretch", height=180)
    st.markdown("</div>", unsafe_allow_html=True)


def render_empty_workbench() -> None:
    with st.expander("更多说明与 CSV 格式", expanded=False):
        st.markdown('<div class="fc3d-panel"><h3>工作台入口</h3>', unsafe_allow_html=True)
        st.markdown(
            """
            <div class="fc3d-grid">
              <div class="fc3d-card"><div class="label">1. 数据导入</div><div class="value">CSV 上传 / 内置数据</div></div>
              <div class="fc3d-card"><div class="label">2. 数据校验</div><div class="value">缺失、重复、日期、号码</div></div>
              <div class="fc3d-card"><div class="label">3. 特征工程</div><div class="value">滚动窗口与历史形态</div></div>
              <div class="fc3d-card"><div class="label">4. 模型训练</div><div class="value">统计基线 + 机器学习</div></div>
              <div class="fc3d-card"><div class="label">5. 时间回测</div><div class="value">固定切分 / 滚动窗口</div></div>
              <div class="fc3d-card"><div class="label">6. 候选输出</div><div class="value">候选数量表格与理由</div></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="fc3d-panel"><h3>CSV 格式</h3>', unsafe_allow_html=True)
        st.dataframe(
            pd.DataFrame(
                [
                    {"issue": "2026001", "date": "2026-01-01", "number": "008"},
                    {"issue": "2026002", "date": "2026-01-02", "number": "521"},
                ]
            ),
            width="stretch",
            hide_index=True,
            height=130,
        )
        st.markdown("</div>", unsafe_allow_html=True)


def render_charts(clean_df: pd.DataFrame) -> None:
    st.markdown('<div class="fc3d-panel"><h3>数据图表</h3>', unsafe_allow_html=True)
    digits = clean_df[["bai", "shi", "ge"]].to_numpy().ravel()
    digit_counts = pd.Series(digits).value_counts().sort_index().reindex(range(10), fill_value=0).reset_index()
    digit_counts.columns = ["digit", "count"]
    sum_counts = clean_df["sum3"].value_counts().sort_index().reset_index()
    sum_counts.columns = ["sum3", "count"]
    pos_counts = pd.DataFrame(
        {
            "百位": clean_df["bai"].value_counts().sort_index().reindex(range(10), fill_value=0),
            "十位": clean_df["shi"].value_counts().sort_index().reindex(range(10), fill_value=0),
            "个位": clean_df["ge"].value_counts().sort_index().reindex(range(10), fill_value=0),
        }
    ).T
    pos_counts.columns = [str(i) for i in range(10)]

    cols = st.columns([1.1, 1.1, 1.0], gap="medium")
    with cols[0]:
        fig = px.bar(digit_counts, x="digit", y="count", title="整体数字频率")
        fig.update_layout(height=300, margin=dict(l=10, r=10, t=38, b=10), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#FFFFFF", font_color="#172033")
        st.plotly_chart(fig, width="stretch")
    with cols[1]:
        fig = px.imshow(pos_counts, text_auto=True, color_continuous_scale=["#EEF2FF", "#93C5FD", "#2563EB"], title="位置频率热力图")
        fig.update_layout(height=300, margin=dict(l=10, r=10, t=38, b=10), paper_bgcolor="rgba(0,0,0,0)", font_color="#172033")
        st.plotly_chart(fig, width="stretch")
    with cols[2]:
        fig = px.bar(sum_counts, x="sum3", y="count", title="和值分布")
        fig.update_layout(height=300, margin=dict(l=10, r=10, t=38, b=10), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#FFFFFF", font_color="#172033")
        st.plotly_chart(fig, width="stretch")
    st.markdown("</div>", unsafe_allow_html=True)


def render_model_controls() -> None:
    st.sidebar.markdown("## 运行控制")
    st.sidebar.radio("主题", THEME_OPTIONS, key="app_theme_mode", horizontal=True)
    uploaded = st.sidebar.file_uploader(
        "上传历史开奖数据CSV（可选）",
        type=["csv"],
        help="有自己的历史开奖数据表时上传 CSV；没有文件也可以直接加载系统缓存的近5年开奖记录。",
    )
    st.sidebar.caption(
        "上传你自己的历史开奖数据表；需要包含 issue期号、date日期、number开奖号，number 要保留前导0。没有CSV也可以直接点击下方按钮加载近5年开奖记录。"
    )
    reload_required = demo_reload_required()
    sidebar_confirm_reload = True
    if reload_required:
        st.sidebar.warning(f"{demo_button_label()}会清空当前训练、回测和候选结果。")
        sidebar_confirm_reload = st.sidebar.checkbox(f"确认{demo_button_label()}", key="sidebar_confirm_reload_demo")
    demo_btn = st.sidebar.button(
        demo_button_label(),
        key="sidebar_load_demo",
        width="stretch",
        disabled=reload_required and not sidebar_confirm_reload,
        type=action_button_type("demo", action_availability()) if not reload_required else "secondary",
        help=disabled_action_reason("demo") if reload_required else "加载系统缓存的近5年开奖记录。",
    )
    update_history_btn = st.sidebar.button(
        "更新近5年缓存",
        key="sidebar_update_history_repository",
        width="stretch",
        help="从在线接口补全本地近5年历史数据缓存。",
    )
    if BUILTIN_HISTORY_PATH.exists():
        st.sidebar.download_button(
            "下载近5年缓存历史CSV",
            data=BUILTIN_HISTORY_PATH.read_bytes(),
            file_name=BUILTIN_HISTORY_PATH.name,
            mime="text/csv",
            width="stretch",
        )
    else:
        st.sidebar.caption("加载近5年开奖记录后，可下载接口缓存 CSV。")

    with st.sidebar.expander("高级设置", expanded=False):
        st.markdown("### 特征窗口")
        st.session_state.window_choices = st.multiselect(
            "滚动窗口",
            options=[30, 60, 120],
            default=DEFAULT_WINDOWS,
        )
        st.markdown("### 回测方式")
        st.session_state.backtest_mode = st.selectbox(
            "时间切分",
            ["固定 80/20", "滚动窗口"],
            index=1,
        )
        st.session_state.train_ratio = st.slider("训练集比例", 0.6, 0.9, 0.8, 0.02)
        st.session_state.rolling_initial_train_size = st.number_input(
            "滚动初始训练窗口",
            min_value=DEFAULT_ROLLING_INITIAL_TRAIN_SIZE,
            max_value=5000,
            value=DEFAULT_ROLLING_INITIAL_TRAIN_SIZE,
            step=50,
            help="滚动回测默认使用前1000期训练，然后按批预测后续期数。",
        )
        st.session_state.rolling_batch_size = st.slider(
            "每批预测期数",
            1,
            100,
            DEFAULT_ROLLING_BATCH_SIZE,
            1,
        )
        st.session_state.top_k = st.slider("候选数量", 10, 50, 20, 5)
        st.session_state.seed = st.number_input("随机种子", min_value=1, max_value=9999, value=42, step=1)

        st.markdown("### 模型家族")
        st.checkbox("统计基线", value=True, disabled=True, help="基线模型始终用于对照。")
        st.session_state.use_history_frequency = st.checkbox("历史频率", value=True)
        st.session_state.use_bayes_smooth_frequency = st.checkbox("贝叶斯平滑", value=True)
        st.session_state.use_omission = st.checkbox("遗漏值", value=True)
        st.session_state.use_sum_distribution = st.checkbox("和值分布", value=True)
        st.session_state.use_position_frequency = st.checkbox("位置频率", value=True)

        st.markdown("#### 传统机器学习")
        st.session_state.use_logistic_regression = st.checkbox("逻辑回归", value=True)
        st.session_state.use_random_forest = st.checkbox("随机森林", value=True)
        st.session_state.use_xgboost = st.checkbox("极端梯度提升", value=False, help="需要已安装 xgboost。")
        st.session_state.use_lightgbm = st.checkbox("轻量梯度提升", value=False, help="需要已安装 lightgbm。")
        st.session_state.use_catboost = st.checkbox("类别提升", value=False, help="需要已安装 catboost。")

        st.markdown("#### 深度学习")
        torch_available = TORCH_AVAILABLE
        st.session_state.use_tabresnet = st.checkbox(
            "残差表格网络",
            value=False,
            disabled=not torch_available,
            help="当前环境已安装 PyTorch，可以启用残差表格网络。" if torch_available else "当前环境未安装 PyTorch，所以暂不能启用残差表格网络。",
        )

        if not torch_available:
            st.caption("残差表格网络已禁用：当前环境未安装 PyTorch。")

        st.markdown("#### 残差表格网络参数")
        st.session_state.torch_epochs = st.number_input("epochs", 1, 100, 12, 1)
        st.session_state.torch_batch_size = st.number_input("batch_size", 8, 256, 32, 8)
        st.session_state.torch_learning_rate = st.number_input("learning_rate", 1e-5, 1e-1, 1e-3, format="%.5f")
        st.session_state.torch_hidden_dim = st.number_input("hidden_dim", 32, 512, 128, 16)
        st.session_state.torch_dropout = st.number_input("dropout", 0.0, 0.8, 0.15, 0.01)
        st.session_state.torch_weight_decay = st.number_input("weight_decay", 0.0, 0.1, 0.0001, format="%.5f")

    st.sidebar.markdown("### 操作")
    availability = action_availability()
    validate_btn = st.sidebar.button(
        "校验数据",
        key="sidebar_validate_data",
        width="stretch",
        disabled=not availability["validate"],
        type=action_button_type("validate", availability),
        help=disabled_action_reason("validate") if not availability["validate"] else "正在校验数据，预计30-40秒。",
    )
    train_btn = st.sidebar.button(
        "训练模型",
        key="sidebar_train_models",
        width="stretch",
        disabled=not availability["train"],
        type=action_button_type("train", availability),
        help=disabled_action_reason("train") if not availability["train"] else "正在训练模型，预计30-60秒。",
    )
    backtest_btn = st.sidebar.button(
        "开始滚动回测",
        key="sidebar_start_backtest",
        width="stretch",
        disabled=not availability["backtest"],
        type=action_button_type("backtest", availability),
        help=disabled_action_reason("backtest") if not availability["backtest"] else "正在滚动回测，预计40-60秒。",
    )
    predict_btn = st.sidebar.button(
        "生成候选与胆码",
        key="sidebar_generate_candidates",
        width="stretch",
        disabled=not availability["predict"],
        type=action_button_type("predict", availability),
        help=disabled_action_reason("predict") if not availability["predict"] else "正在生成候选，请稍候。",
    )
    report_btn = st.sidebar.button(
        "导出分析报告",
        key="sidebar_export_report",
        width="stretch",
        disabled=not availability["export"],
        type=action_button_type("export", availability),
        help=disabled_action_reason("export") if not availability["export"] else "导出完整分析报告。",
    )

    return uploaded, demo_btn, update_history_btn, validate_btn, train_btn, backtest_btn, predict_btn, report_btn


def weight_display_df() -> pd.DataFrame:
    weights = st.session_state.backtest_weights or {}
    rows = [
        {"模型": display_model_name(k), "归一化权重": f"{v:.3f}"}
        for k, v in sorted(weights.items(), key=lambda item: item[1], reverse=True)
    ]
    return pd.DataFrame(rows)


def key_weight_items(limit: int = 3) -> list[tuple[str, float]]:
    weights = st.session_state.backtest_weights or {}
    if not weights:
        return []
    return [(name, float(value)) for name, value in sorted(weights.items(), key=lambda item: item[1], reverse=True)[:limit]]


def render_weight_panel(title: str = "当前权重") -> None:
    st.markdown(f'<div class="fc3d-panel"><h3>{title}</h3>', unsafe_allow_html=True)
    weight_df = weight_display_df()
    key_items = key_weight_items()
    if not weight_df.empty:
        weights = st.session_state.backtest_weights or {}
        weight_sum = sum(float(value) for value in weights.values())
        enabled_text = display_model_list(list(weights.keys()))
        summary_cards = "".join(
            (
                '<div class="fc3d-weight-pill">'
                f'<div class="label">{escape(display_model_name(name))}</div>'
                f'<div class="value">{value:.3f}</div>'
                "</div>"
            )
            for name, value in key_items
        )
        if summary_cards:
            st.markdown(
                f'<div class="fc3d-note" style="margin-bottom:0.35rem">当前启用模型：{escape(enabled_text)}。归一化权重总和 = {weight_sum:.3f}。</div><div class="fc3d-weight-summary">{summary_cards}</div>',
                unsafe_allow_html=True,
            )
        st.markdown('<div class="fc3d-note" style="margin-bottom:0.45rem">模型输出已做尺度校准与平滑，避免单个模型分数过度主导；这不是概率校准，模型相对分数只影响排序，不是中奖概率。</div>', unsafe_allow_html=True)
        with st.expander("查看全部模型权重", expanded=False):
            st.dataframe(weight_df, width="stretch", hide_index=True)
    else:
        st.markdown('<div class="fc3d-note">完成回测后，这里会显示模型权重。未回测时生成候选会使用等权集成。</div>', unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


def render_right_rail() -> None:
    report = st.session_state.validation_report
    st.markdown('<div class="fc3d-panel"><h3>状态面板</h3>', unsafe_allow_html=True)
    state: AppState = st.session_state.app_state
    status_items = [
        ("当前状态", state.current_mode),
        ("最后加载", state.last_loaded_at or "—"),
        ("最后训练", state.last_trained_at or "—"),
        ("最后回测", state.last_backtest_at or "—"),
        ("最后候选生成", state.last_predict_at or "—"),
    ]
    for label, value in status_items:
        st.markdown(
            f"""
            <div class="fc3d-mini" style="margin-bottom:0.5rem">
              <div class="label">{label}</div>
              <div class="value">{value}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    if report:
        if report["warning_small_sample"]:
            st.markdown(
                '<div class="fc3d-warning">当前样本量较少，模型结果仅适合学习和模拟研究。建议准备 1000 期以上数据后再观察长期稳定性。</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="fc3d-good">样本量已可支撑基础训练和严格回测，但仍要和理论随机基线比较。</div>',
                unsafe_allow_html=True,
            )
    st.markdown("</div>", unsafe_allow_html=True)

    render_weight_panel("当前权重")

    st.markdown('<div class="fc3d-panel"><h3>运行日志</h3>', unsafe_allow_html=True)
    if st.session_state.logs:
        st.code("\n".join(st.session_state.logs[-18:]), language="text")
    else:
        st.markdown('<div class="fc3d-note">这里会显示校验耗时、训练耗时、使用模型、回测样本数量、缺依赖跳过模型和候选生成记录。</div>', unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="fc3d-panel"><h3>说明</h3>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="fc3d-note">
          1. 先上传历史开奖 CSV，或点击侧边栏按钮加载近5年开奖记录，然后校验。<br>
          2. 再训练和回测，观察是否明显超过理论随机基线。<br>
          3. 候选号码只是排序结果，不是预测承诺。<br>
          4. 如果模型没有显著超过理论随机基线，说明历史数据里没有稳定可学习规律。
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)


def format_candidate_group(candidate_df: pd.DataFrame) -> pd.DataFrame:
    if candidate_df is None or candidate_df.empty:
        return pd.DataFrame(columns=["排名", "号码", "模型相对分数", "支持模型", "理由"])
    show_df = candidate_df[["rank", "number", "score", "source_models", "reason"]].copy()
    show_df["score"] = show_df["score"].map(lambda value: f"{float(value):.4f}")
    return show_df.rename(
        columns={
            "rank": "排名",
            "number": "号码",
            "score": "模型相对分数",
            "source_models": "支持模型",
            "reason": "理由",
        }
    )


def format_343_export_dataframe(pool_df: pd.DataFrame) -> pd.DataFrame:
    if pool_df is None or pool_df.empty:
        return pd.DataFrame(columns=["rank", "number", "model_relative_score", "source_models", "reason"])
    export_df = pool_df[["rank", "number", "score", "source_models", "reason"]].copy()
    export_df = export_df.rename(columns={"score": "model_relative_score"})
    return export_df


def candidate_model_context_dataframe(
    model_scores: dict[str, np.ndarray] | None,
    backtest_metrics: dict[str, dict[str, float]] | None,
    weights: dict[str, float] | None,
) -> pd.DataFrame:
    active_models = display_model_list(list(active_score_items(model_scores or {}).keys()))
    if active_models == "无":
        active_models = "尚未生成候选"

    metrics = backtest_metrics or {}
    best_position = best_metric(metrics, "position_top7_all_hit_rate")
    best_top10 = best_metric(metrics, "top10_hit_rate")
    if best_position and best_top10:
        best_text = (
            f"三位同时Top7：{display_model_name(best_position[0])} "
            f"{float(best_position[1].get('position_top7_all_hit_rate', 0.0)):.3%}；"
            f"直选Top10：{display_model_name(best_top10[0])} "
            f"{float(best_top10[1].get('top10_hit_rate', 0.0)):.3%}"
        )
    else:
        best_text = "尚未完成回测，暂不能判断最佳模型"

    if weights:
        weight_sum = sum(float(value) for value in weights.values())
        top_weight_text = "、".join(
            f"{display_model_name(name)} {float(value):.3f}"
            for name, value in sorted(weights.items(), key=lambda item: item[1], reverse=True)[:3]
        )
        weight_text = f"回测权重，归一化总和 {weight_sum:.3f}；前三权重：{top_weight_text}"
    else:
        weight_text = "等权集成：尚无回测权重或未启用权重收缩"

    return pd.DataFrame(
        [
            {"项目": "当前候选使用模型", "内容": active_models},
            {"项目": "回测最佳模型", "内容": best_text},
            {"项目": "最终权重来源", "内容": weight_text},
        ]
    )


def compact_candidate_grid_html(candidate_df: pd.DataFrame) -> str:
    if candidate_df is None or candidate_df.empty:
        return '<div class="fc3d-empty-candidates">暂无可展示候选</div>'
    cards: list[str] = []
    for idx, row in enumerate(candidate_df.head(20).itertuples(index=False), start=1):
        rank = escape(str(getattr(row, "rank", idx)))
        number = escape(str(getattr(row, "number", "")))
        score = float(getattr(row, "score", 0.0))
        models = escape(str(getattr(row, "source_models", "")))
        cards.append(
            '<div class="fc3d-compact-candidate">'
            f'<div class="rank">#{rank}</div>'
            f'<div class="number">{number}</div>'
            f'<div class="score">模型相对分数 {score:.4f}</div>'
            f'<div class="models">{models}</div>'
            '</div>'
        )
    return f'<div class="fc3d-compact-candidate-grid">{"".join(cards)}</div>'


def candidate_group_html(candidate_df: pd.DataFrame) -> str:
    if candidate_df is None or candidate_df.empty:
        return '<div class="fc3d-empty-candidates">暂无可展示候选</div>'
    rows: list[str] = []
    for idx, row in enumerate(candidate_df.head(20).itertuples(index=False), start=1):
        rank = escape(str(getattr(row, "rank", idx)))
        number = escape(str(getattr(row, "number", "")))
        score = float(getattr(row, "score", 0.0))
        models = escape(str(getattr(row, "source_models", "")))
        rows.append(
            '<div class="fc3d-candidate-row">'
            f'<div class="rank">#{rank}</div>'
            f'<div class="number">{number}</div>'
            '<div class="meta">'
            f'<b>模型相对分数 {score:.4f}</b><br>'
            f'{models}'
            '</div>'
            '</div>'
        )
    return f'<div class="fc3d-candidate-list">{"".join(rows)}</div>'


def render_candidate_group_table(label: str, note: str, candidate_df: pd.DataFrame, tone: str = "") -> None:
    tone_class = f" {tone}" if tone else ""
    st.markdown(
        f'<div class="fc3d-candidate-section{tone_class}"><h4>{escape(label)}</h4>'
        f'<div class="fc3d-note" style="margin-bottom:0.45rem">{escape(note)}</div>',
        unsafe_allow_html=True,
    )
    st.markdown(compact_candidate_grid_html(candidate_df), unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


def render_candidate_table(candidate_df: pd.DataFrame) -> None:
    render_danma_module(current_danma_prediction(), st.session_state.backtest_metrics or {}, expanded_ranking=False)
    position_report = st.session_state.get("position_7_report")
    pool_df = st.session_state.get("candidate_pool_df")
    candidate_groups = st.session_state.get("candidate_groups") or {}
    main_df = candidate_groups.get("main", candidate_df)
    diverse_df = candidate_groups.get("diverse", pd.DataFrame())
    long_tail_df = candidate_groups.get("long_tail", pd.DataFrame())
    render_position_7_module(position_report, st.session_state.backtest_metrics or {})
    st.markdown('<div class="fc3d-panel"><h3>候选号码分组</h3>', unsafe_allow_html=True)
    st.markdown(f'<div class="fc3d-note">{SIMULATION_TEXT} {SAFETY_TEXT}</div>', unsafe_allow_html=True)
    st.markdown('<div class="fc3d-note" style="margin-top:0.35rem">主候选按集成模型相对分数排序；分散候选尽量拉开号码结构；冷门参考来自343组候选池后段，仅用于观察分布。</div>', unsafe_allow_html=True)
    st.markdown('<div class="fc3d-candidate-section"><h4>主候选 Top20</h4>', unsafe_allow_html=True)
    st.markdown(compact_candidate_grid_html(main_df), unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
    render_candidate_group_table(
        "分散候选 Top20",
        "从同一个343组候选池中挑选，尽量减少与已选号码的逐位重复。",
        diverse_df,
        tone="diverse",
    )
    render_candidate_group_table(
        "冷门参考 Top20",
        "来自343组候选池的后段低分区域，只作分布参考，不代表更可能出现。",
        long_tail_df,
        tone="long-tail",
    )

    st.markdown('<h4 style="margin:0.9rem 0 0.35rem 0">0-9完整排名和模型支持来源</h4>', unsafe_allow_html=True)
    context_df = candidate_model_context_dataframe(
        st.session_state.get("candidate_model_scores") or st.session_state.get("trained_models"),
        st.session_state.get("backtest_metrics"),
        st.session_state.get("backtest_weights"),
    )
    st.dataframe(context_df, width="stretch", hide_index=True)
    render_no_position_7_module(current_no_position_7_prediction(), st.session_state.backtest_metrics or {}, expanded_ranking=False)

    report_text = build_report_text(st.session_state.validation_report, st.session_state.backtest_metrics, main_df)
    ensure_report_download(report_text)
    st.markdown('<div class="fc3d-export-note">候选结果已生成，可直接导出当前数据、回测结论和候选号码。</div>', unsafe_allow_html=True)
    st.download_button(
        "导出完整分析报告",
        data=report_text,
        file_name=st.session_state.last_report_file,
        mime="text/markdown",
        key="candidate_export_report",
        width="stretch",
        on_click="ignore",
    )
    if pool_df is not None and not pool_df.empty:
        export_pool = format_343_export_dataframe(pool_df)
        st.download_button(
            "导出全部343组合",
            data=export_pool.to_csv(index=False).encode("utf-8-sig"),
            file_name=f"fc3d_343_pool_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            key="candidate_export_343_pool",
            width="stretch",
            on_click="ignore",
        )

    render_weight_panel("模型权重")

    with st.container(key="candidate_wide_table"):
        st.dataframe(
            format_candidate_group(main_df),
            width="stretch",
            hide_index=True,
            key="candidate_wide_table_frame",
            column_config={
                "排名": st.column_config.NumberColumn(width="small"),
                "号码": st.column_config.TextColumn(width="small"),
                "模型相对分数": st.column_config.TextColumn(width="small"),
                "支持模型": st.column_config.TextColumn(width="medium"),
                "理由": st.column_config.TextColumn(width="large"),
            },
        )
    with st.expander("每个候选号码的模型支持分数", expanded=False):
        for row in main_df.head(10).itertuples(index=False):
            support = row.support if isinstance(row.support, dict) else {}
            support_text = " / ".join(
                f"{escape(display_model_name(name))}:{float(value):.4f}"
                for name, value in sorted(support.items(), key=lambda item: item[1], reverse=True)[:5]
                if name != "random_baseline"
            )
            rank = escape(str(getattr(row, "rank", "")))
            number = escape(str(getattr(row, "number", "")))
            source_models = escape(str(getattr(row, "source_models", "")))
            reason = escape(str(getattr(row, "reason", "")))
            st.markdown(
                f"""
                <div class="fc3d-mini" style="margin-bottom:0.55rem">
                  <div class="value">{rank}. {number} | 模型相对分数 {float(row.score):.4f}</div>
                  <div class="label">支持模型：{source_models}</div>
                  <div class="label">模型分数：{support_text}</div>
                  <div class="label">理由：{reason}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
    st.markdown("</div>", unsafe_allow_html=True)


def render_candidate_placeholder() -> None:
    render_danma_module(current_danma_prediction(), st.session_state.backtest_metrics or {}, expanded_ranking=False)
    render_position_7_module(None, st.session_state.backtest_metrics or {})
    st.markdown('<div class="fc3d-panel"><h3>Top 候选号码</h3>', unsafe_allow_html=True)
    st.markdown(f'<div class="fc3d-note">{SIMULATION_TEXT} 上传并校验数据后，可生成候选表。</div>', unsafe_allow_html=True)
    st.table(
        pd.DataFrame(
            [
                {
                    "排名": "—",
                    "号码": "待生成",
                    "模型相对分数": "—",
                    "和值": "—",
                    "跨度": "—",
                    "形态": "—",
                    "来源模型": "等待训练或候选生成",
                    "理由": "上传并校验数据后显示",
                }
            ]
        )
    )
    st.markdown("</div>", unsafe_allow_html=True)
    render_no_position_7_module(current_no_position_7_prediction(), st.session_state.backtest_metrics or {}, expanded_ranking=False)


def render_backtest_results(metrics: dict[str, dict[str, float]], note: str, fold_df: pd.DataFrame | None = None) -> None:
    st.markdown('<div class="fc3d-panel"><h3>回测结果</h3>', unsafe_allow_html=True)
    result_note = "回测已完成，以下为时间顺序回测结果。" if metrics else note
    if metrics and note and note != result_note:
        result_note = f"{result_note} {note}"
    st.markdown(
        f'<div class="fc3d-note">{escape(result_note)}</div>',
        unsafe_allow_html=True,
    )
    st.markdown(backtest_conclusion_html(metrics, note, fold_df), unsafe_allow_html=True)
    verdict_text, verdict_class = baseline_comparison_text(metrics)
    summary_verdict_df = baseline_summary_dataframe(metrics)
    if not summary_verdict_df.empty:
        statistic_row = summary_verdict_df[summary_verdict_df["项目"] == "统计结论"]
        statistic_text = statistic_row.iloc[0]["结论"] if not statistic_row.empty else "未达到显著优势"
        st.markdown(
            f'<div class="{verdict_class}" style="margin-bottom:0.55rem"><b>统计结论：</b>{escape(str(statistic_text))}</div>',
            unsafe_allow_html=True,
        )
        st.dataframe(summary_verdict_df, width="stretch", hide_index=True)
    table = metrics_table(metrics)
    position_table = position_metrics_table(metrics)
    if metrics:
        summary_df = backtest_run_summary_dataframe(metrics, note, fold_df)
        if not summary_df.empty:
            st.dataframe(summary_df, width="stretch", hide_index=True)
        summary_html = position_metric_summary_html(metrics)
        if summary_html:
            st.markdown(summary_html, unsafe_allow_html=True)
        st.markdown('<div class="fc3d-note" style="margin-bottom:0.4rem">位置级Top7回测</div>', unsafe_allow_html=True)
        st.dataframe(position_table, width="stretch", hide_index=True)
        st.markdown('<div class="fc3d-note" style="margin:0.7rem 0 0.4rem 0">直选组合回测：Top10 / Top20 / Top50 分表</div>', unsafe_allow_html=True)
        direct_cols = st.columns(3, gap="small")
        for col, label in zip(direct_cols, ["Top10", "Top20", "Top50"]):
            with col:
                st.markdown(
                    f'<div class="fc3d-direct-table-wrap"><div class="fc3d-note" style="margin-bottom:0.2rem">{label}</div>'
                    '<div class="fc3d-scroll-hint">表格可横向滚动</div>',
                    unsafe_allow_html=True,
                )
                st.dataframe(direct_metric_table(metrics, label), width="stretch", height=250, hide_index=True)
                st.markdown("</div>", unsafe_allow_html=True)
        audit_table = backtest_audit_table(metrics)
        if not audit_table.empty:
            st.markdown(
                '<div class="fc3d-note" style="margin:0.7rem 0 0.4rem 0">统计审计：命中次数 / 理论期望 / 差值 / 95%置信区间 / 近似p值 / 是否显著</div>',
                unsafe_allow_html=True,
            )
            st.markdown(significance_badges_html(audit_table), unsafe_allow_html=True)
            st.dataframe(audit_table, width="stretch", hide_index=True)
        danma_audit = danma_backtest_audit_table(metrics)
        if not danma_audit.empty:
            st.markdown(
                f'<div class="fc3d-note" style="margin:0.7rem 0 0.4rem 0">胆码回测：{danma_statistical_conclusion(metrics)}</div>',
                unsafe_allow_html=True,
            )
            st.dataframe(danma_backtest_summary_table(metrics), width="stretch", hide_index=True)
            st.dataframe(danma_audit, width="stretch", hide_index=True)
            if danma_statistical_conclusion(metrics) != "统计结论：达到显著优势":
                st.markdown(f'<div class="fc3d-warning">{DANMA_INEFFECTIVE_TEXT}</div>', unsafe_allow_html=True)
        no_position_7_audit = no_position_7_backtest_audit_table(metrics)
        if not no_position_7_audit.empty:
            st.markdown(
                f'<div class="fc3d-note" style="margin:0.7rem 0 0.4rem 0">不定位7码回测：{no_position_7_statistical_conclusion(metrics)}</div>',
                unsafe_allow_html=True,
            )
            st.dataframe(no_position_7_backtest_summary_table(metrics), width="stretch", hide_index=True)
            st.dataframe(no_position_7_audit, width="stretch", hide_index=True)
    else:
        st.table(
            pd.DataFrame(
                [
                    {
                        "模型": "待回测",
                        "样本": "—",
                        "百位Top7": "—",
                        "十位Top7": "—",
                        "个位Top7": "—",
                        "三位同入7码": "—",
                        "Top10": "—",
                        "Top20": "—",
                        "Top50": "—",
                    }
                ]
            )
        )
    if metrics and "模型" in table.columns:
        chart_df = table[
            table["模型"].isin(
                [
                    "理论随机基线",
                    "历史频率",
                    "贝叶斯平滑",
                    "位置频率",
                    "随机森林",
                    "逻辑回归",
                    "残差表格网络",
                    "等权集成",
                ]
            )
        ]
        if not chart_df.empty:
            chart = px.bar(chart_df, x="模型", y="Top10", color="模型", title="Top10 命中率对比")
            chart.update_layout(height=320, margin=dict(l=10, r=10, t=40, b=10), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#FFFFFF", font_color="#172033", showlegend=False)
            st.plotly_chart(chart, width="stretch")
    if fold_df is not None and not fold_df.empty:
        with st.expander("滚动折详情", expanded=False):
            st.dataframe(fold_df, width="stretch", hide_index=True)
        top10_cols = [c for c in fold_df.columns if c.endswith("_top10")]
        if top10_cols:
            summary_rows = []
            for col in top10_cols:
                model = col.replace("_top10", "")
                series = fold_df[col].astype(float)
                summary_rows.append(
                    {
                        "模型": display_model_name(model),
                        "Top10均值": float(series.mean()),
                        "Top10标准差": float(series.std(ddof=0)),
                    }
                )
            with st.expander("滚动均值 / 波动", expanded=False):
                st.dataframe(pd.DataFrame(summary_rows).sort_values("模型"), width="stretch", hide_index=True)
    st.markdown(
        f'<div class="fc3d-warning" style="margin-top:0.6rem">{RANDOM_BASELINE_TEXT}</div>',
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)


def ensure_report_download(report_text: str) -> None:
    st.session_state.last_report_text = report_text
    st.session_state.last_report_file = f"fc3d_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"


def candidate_and_danma_export_dataframe() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    danma_prediction = current_danma_prediction()
    if danma_prediction:
        ranking = danma_prediction.get("ranking")
        if ranking is not None and not ranking.empty:
            for row in ranking.head(3).to_dict("records"):
                rows.append(
                    {
                        "类型": "下一期3个胆码",
                        "排名": int(row.get("rank", 0)),
                        "号码": str(row.get("digit", "")),
                        "模型相对分数": f"{float(row.get('score', 0.0)):.6f}",
                        "支持模型": str(row.get("source_models", "")),
                        "理由": "胆码模型排序结果",
                    }
                )
    candidate_groups = st.session_state.get("candidate_groups") or {}
    candidate_sections = [
        ("主候选Top20", candidate_groups.get("main", st.session_state.get("candidate_df"))),
        ("分散候选Top20", candidate_groups.get("diverse", pd.DataFrame())),
        ("冷门参考Top20", candidate_groups.get("long_tail", pd.DataFrame())),
    ]
    for section_name, data in candidate_sections:
        if data is None or data.empty:
            continue
        for idx, row in enumerate(data.head(20).itertuples(index=False), start=1):
            rows.append(
                {
                    "类型": section_name,
                    "排名": int(getattr(row, "rank", idx)),
                    "号码": str(getattr(row, "number", "")),
                    "模型相对分数": f"{float(getattr(row, 'score', 0.0)):.6f}",
                    "支持模型": str(getattr(row, "source_models", "")),
                    "理由": str(getattr(row, "reason", "")),
                }
            )
    return pd.DataFrame(rows, columns=["类型", "排名", "号码", "模型相对分数", "支持模型", "理由"])


def render_export_center(report_text: str) -> None:
    report = st.session_state.validation_report or {}
    data_version = str(report.get("data_version", st.session_state.app_state.data_version or "—"))
    has_backtest = bool(st.session_state.backtest_metrics)
    has_candidates = st.session_state.candidate_df is not None
    has_position_7 = st.session_state.get("position_7_report") is not None
    has_danma = current_danma_prediction() is not None
    content_items = [
        ("数据版本", data_version),
        ("回测结论", "已包含" if has_backtest else "待回测"),
        ("7码结果", "已包含" if has_position_7 else "待生成"),
        ("下一期3个胆码", "已包含" if has_danma else "待生成"),
        ("主候选Top20", "已包含" if has_candidates else "待生成"),
        ("Top10/Top20/Top50回测", "已包含" if has_backtest else "待回测"),
        ("统计审计", "已包含" if has_backtest else "待回测"),
    ]
    cards = "".join(
        (
            '<div class="fc3d-mini">'
            f'<div class="label">{escape(label)}</div>'
            f'<div class="value">{escape(value)}</div>'
            "</div>"
        )
        for label, value in content_items
    )
    st.markdown(
        f"""
        <div class="fc3d-panel">
          <h3>报告中心</h3>
          <div class="fc3d-note">当前状态：{escape(app_stage())}。报告只汇总页面已生成的模拟分析结果，不新增任何模型计算。</div>
          <h4 style="margin:0.8rem 0 0.35rem 0">报告包含内容</h4>
          <div class="fc3d-export-center-grid">{cards}</div>
        """,
        unsafe_allow_html=True,
    )
    if not has_backtest or not has_candidates:
        st.markdown(
            '<div class="fc3d-warning" style="margin-bottom:0.6rem">当前报告尚未包含完整回测和候选结果，请先完成时间顺序回测并生成候选。</div>',
            unsafe_allow_html=True,
        )
    st.download_button(
        "导出完整分析报告",
        data=report_text,
        file_name=st.session_state.last_report_file,
        mime="text/markdown",
        key="export_full_report",
        width="stretch",
    )
    pool_df = st.session_state.get("candidate_pool_df")
    if pool_df is not None and not pool_df.empty:
        export_pool = format_343_export_dataframe(pool_df)
        st.download_button(
            "导出全部343组合",
            data=export_pool.to_csv(index=False).encode("utf-8-sig"),
            file_name=f"fc3d_343_pool_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            key="export_tab_343_pool",
            width="stretch",
        )
    else:
        st.download_button(
            "导出全部343组合",
            data=b"",
            file_name="fc3d_343_pool_pending.csv",
            mime="text/csv",
            key="export_tab_343_pool_disabled",
            width="stretch",
            disabled=True,
            help="请先生成候选与胆码。",
        )
    candidate_export_df = candidate_and_danma_export_dataframe()
    st.download_button(
        "导出胆码与候选CSV",
        data=candidate_export_df.to_csv(index=False).encode("utf-8-sig"),
        file_name=f"fc3d_danma_candidates_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
        key="export_danma_candidates",
        width="stretch",
        disabled=candidate_export_df.empty,
        help="请先生成候选与胆码。" if candidate_export_df.empty else "导出当前页面展示的胆码、主候选、分散候选和冷门参考。",
    )
    with st.expander("查看报告预览", expanded=False):
        st.code(report_text, language="markdown")
    st.markdown("</div>", unsafe_allow_html=True)


def backtest_run_summary_dataframe(metrics: dict[str, dict[str, float]], note: str, fold_df: pd.DataFrame | None = None) -> pd.DataFrame:
    if not metrics:
        return pd.DataFrame(columns=["项目", "值"])
    mode = "滚动回测" if "滚动回测" in note else "固定切分"
    test_samples = max((int(metric.get("samples", 0)) for name, metric in metrics.items() if name != "random_baseline"), default=0)
    train_match = re.search(r"训练样本数量\s*(\d+)", note)
    test_match = re.search(r"测试样本数量\s*(\d+)", note)
    window_match = re.search(r"滚动窗口数量\s*(\d+|不适用)", note)
    train_samples = train_match.group(1) if train_match else "—"
    if test_match:
        test_samples_value = test_match.group(1)
    else:
        test_samples_value = str(test_samples)
    if fold_df is not None and not fold_df.empty:
        rolling_windows = str(len(fold_df))
    elif window_match:
        rolling_windows = window_match.group(1)
    else:
        rolling_windows = "不适用" if mode == "固定切分" else "0"
    return pd.DataFrame(
        [
            {"项目": "回测模式", "值": mode},
            {"项目": "训练样本数量", "值": train_samples},
            {"项目": "测试样本数量", "值": test_samples_value},
            {"项目": "滚动窗口数量", "值": rolling_windows},
        ]
    )


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, page_icon="3D", layout="wide", initial_sidebar_state="collapsed")
    init_state()
    css()
    render_header()

    uploaded, sidebar_demo_btn, sidebar_update_history_btn, sidebar_validate_btn, sidebar_train_btn, sidebar_backtest_btn, sidebar_predict_btn, sidebar_report_btn = render_model_controls()

    if uploaded is not None:
        raw_bytes = uploaded.getvalue()
        if st.session_state.raw_bytes != raw_bytes:
            st.session_state.raw_df = None
            st.session_state.clean_df = None
            st.session_state.validation_report = None
            st.session_state.feature_df = None
            st.session_state.backtest_metrics = None
            st.session_state.backtest_weights = None
            st.session_state.backtest_note = ""
            st.session_state.backtest_fold_df = None
            st.session_state.candidate_df = None
            st.session_state.candidate_groups = None
            st.session_state.candidate_pool_df = None
            st.session_state.candidate_model_scores = None
            st.session_state.position_7_report = None
            st.session_state.danma_prediction = None
            st.session_state.no_position_7_prediction = None
        st.session_state.raw_bytes = raw_bytes
        st.session_state.app_state.file_name = uploaded.name
        st.session_state.app_state.data_version = short_hash(raw_bytes)
        st.session_state.app_state.current_mode = "已加载未校验"

    report = st.session_state.validation_report
    sync_current_mode()
    topbar_slot = st.empty()
    with topbar_slot.container():
        render_topbar(report)
    main_demo_btn, main_validate_btn, main_train_btn, main_backtest_btn, main_predict_btn, main_report_btn = render_primary_actions(report)
    render_latest_summary()

    demo_btn = sidebar_demo_btn or main_demo_btn
    validate_btn = sidebar_validate_btn or main_validate_btn
    train_btn = sidebar_train_btn or main_train_btn
    backtest_btn = sidebar_backtest_btn or main_backtest_btn
    predict_btn = sidebar_predict_btn or main_predict_btn
    report_btn = sidebar_report_btn or main_report_btn

    if demo_btn:
        st.session_state.active_tab = "数据"
    if sidebar_update_history_btn:
        st.session_state.active_tab = "数据"
    if validate_btn:
        st.session_state.active_tab = "数据"
    if train_btn or backtest_btn:
        st.session_state.active_tab = "回测"
    if predict_btn:
        st.session_state.active_tab = "候选与胆码"
    if report_btn:
        st.session_state.active_tab = "导出"

    if demo_btn and uploaded is None:
        try:
            demo_df, demo_file_name, demo_source_name = load_default_history()
        except Exception as exc:
            st.error(f"获取近5年开奖记录失败：{exc}")
            log_event(f"获取近5年开奖记录失败：{exc}")
            demo_df = None
        if demo_df is None:
            return
        demo_bytes = demo_df.to_csv(index=False).encode("utf-8-sig")
        st.session_state.raw_bytes = demo_bytes
        st.session_state.raw_df = None
        st.session_state.clean_df = None
        st.session_state.validation_report = None
        st.session_state.feature_df = None
        st.session_state.backtest_metrics = None
        st.session_state.backtest_weights = None
        st.session_state.backtest_note = ""
        st.session_state.backtest_fold_df = None
        st.session_state.candidate_df = None
        st.session_state.candidate_groups = None
        st.session_state.candidate_pool_df = None
        st.session_state.candidate_model_scores = None
        st.session_state.position_7_report = None
        st.session_state.danma_prediction = None
        st.session_state.no_position_7_prediction = None
        st.session_state.app_state.file_name = demo_file_name
        st.session_state.app_state.data_version = short_hash(demo_bytes)
        st.session_state.app_state.last_loaded_at = now_str()
        st.session_state.app_state.current_mode = f"{demo_source_name}已加载"
        st.session_state.main_confirm_reload_demo = False
        st.session_state.sidebar_confirm_reload_demo = False
        log_event(f"已加载{demo_source_name}：{demo_file_name}。")
        st.rerun()

    if sidebar_update_history_btn:
        with st.spinner("正在通过在线接口获取最新数据并补全本地数据仓库..."):
            try:
                update_result = update_history_repository(BUILTIN_HISTORY_PATH)
                updated_df = read_csv_bytes(BUILTIN_HISTORY_PATH.read_bytes())
                updated_bytes = updated_df.to_csv(index=False).encode("utf-8-sig")
                st.session_state.raw_bytes = updated_bytes
                st.session_state.raw_df = None
                st.session_state.clean_df = None
                st.session_state.validation_report = None
                st.session_state.feature_df = None
                st.session_state.trained_models = None
                st.session_state.backtest_metrics = None
                st.session_state.backtest_weights = None
                st.session_state.backtest_note = ""
                st.session_state.backtest_fold_df = None
                st.session_state.candidate_df = None
                st.session_state.candidate_groups = None
                st.session_state.candidate_pool_df = None
                st.session_state.candidate_model_scores = None
                st.session_state.position_7_report = None
                st.session_state.danma_prediction = None
                st.session_state.no_position_7_prediction = None
                st.session_state.app_state.file_name = f"{BUILTIN_HISTORY_PATH.name}（本地数据仓库）"
                st.session_state.app_state.data_version = short_hash(updated_bytes)
                st.session_state.app_state.last_loaded_at = now_str()
                st.session_state.app_state.current_mode = "最新数据已获取"
                if int(update_result["added_count"]) > 0:
                    st.success(
                        f"已补全 {int(update_result['added_count'])} 期，"
                        f"最新 {update_result['latest_issue']} / {update_result['latest_date']} / {update_result['latest_number']}。"
                    )
                else:
                    st.info(
                        f"本地数据仓库已是最新，最新 {update_result['latest_issue']} / "
                        f"{update_result['latest_date']} / {update_result['latest_number']}。"
                    )
                log_event(
                    f"获取最新数据：来源 {update_result['source_name']}，"
                    f"补全 {update_result['added_count']} 期，当前 {update_result['row_count']} 期。"
                )
                st.rerun()
            except Exception as exc:
                st.error(f"获取最新数据失败：{exc}")
                log_event(f"获取最新数据失败：{exc}")

    if validate_btn:
        if st.session_state.raw_bytes is None:
            st.warning("请先上传 CSV 或加载内置数据。")
        else:
            with st.spinner("正在校验数据，预计30-40秒..."):
                try:
                    started_at = time.perf_counter()
                    raw_df = read_csv_bytes(st.session_state.raw_bytes)
                    st.session_state.raw_df = raw_df
                    clean_df, report = inspect_and_clean(raw_df, st.session_state.app_state.file_name, st.session_state.raw_bytes)
                    core = get_core()
                    st.session_state.clean_df = clean_df
                    st.session_state.validation_report = report
                    st.session_state.feature_df = core.build_feature_table(clean_df, windows=selected_windows())
                    st.session_state.trained_models = None
                    st.session_state.backtest_metrics = None
                    st.session_state.backtest_weights = None
                    st.session_state.backtest_note = ""
                    st.session_state.backtest_fold_df = None
                    st.session_state.candidate_df = None
                    st.session_state.candidate_groups = None
                    st.session_state.candidate_pool_df = None
                    st.session_state.candidate_model_scores = None
                    st.session_state.position_7_report = None
                    st.session_state.danma_prediction = None
                    st.session_state.no_position_7_prediction = None
                    st.session_state.app_state.last_loaded_at = now_str()
                    st.session_state.app_state.current_mode = "已校验"
                    log_event(f"完成数据校验：原始 {report['raw_rows']} 期，保留 {report['kept_rows']} 期；校验耗时 {time.perf_counter() - started_at:.2f} 秒。")
                    st.rerun()
                except Exception as exc:
                    st.session_state.validation_report = None
                    st.session_state.clean_df = None
                    st.session_state.feature_df = None
                    st.session_state.candidate_df = None
                    st.session_state.candidate_groups = None
                    st.session_state.candidate_pool_df = None
                    st.session_state.position_7_report = None
                    st.session_state.danma_prediction = None
                    st.session_state.no_position_7_prediction = None
                    st.error(f"数据校验失败：{exc}")
                    log_event(f"数据校验失败：{exc}")

    report = st.session_state.validation_report
    sync_current_mode()
    clean_df = st.session_state.clean_df
    feature_df = st.session_state.feature_df

    main_col, right_col = st.columns([3.35, 1.25], gap="large")
    with main_col:
        if report:
            render_data_cards(report)
            render_validation_panel(report)

        active_tab = st.radio(
            "工作区",
            ["数据", "回测", "候选与胆码", "导出"],
            key="active_tab",
            horizontal=True,
            help="按流程查看数据、回测、候选与胆码、导出。",
        )

        if active_tab == "数据":
            st.markdown(
                f'<div class="fc3d-panel"><h3>数据工作区</h3><div class="fc3d-note">当前状态：{escape(app_stage())}。未加载数据时，请点击“加载近5年开奖记录”或上传CSV；已加载后请点击“校验数据”。</div></div>',
                unsafe_allow_html=True,
            )
            if clean_df is not None and not clean_df.empty:
                render_charts(clean_df)
                st.markdown('<div class="fc3d-panel"><h3>最近开奖记录</h3>', unsafe_allow_html=True)
                st.dataframe(clean_df.tail(20), width="stretch", hide_index=True)
                st.markdown("</div>", unsafe_allow_html=True)
            else:
                render_empty_workbench()

        if active_tab == "回测":
            st.markdown(
                f'<div class="fc3d-panel"><h3>回测工作区</h3><div class="fc3d-note">当前状态：{escape(app_stage())}。已训练后点击“开始滚动回测”，重点看统计结论和是否显著。</div></div>',
                unsafe_allow_html=True,
            )
            if train_btn and clean_df is not None and feature_df is not None:
                with st.spinner("正在训练模型，预计30-60秒..."):
                    try:
                        started_at = time.perf_counter()
                        fit_deep = bool(st.session_state.use_tabresnet)
                        skipped = skipped_model_reasons(fit_deep, train_samples=len(feature_df))
                        model_scores = train_full_model_scores(
                            clean_df=clean_df,
                            feature_df=feature_df,
                            seed=int(st.session_state.seed),
                            fit_deep=fit_deep,
                        )
                        st.session_state.trained_models = model_scores
                        st.session_state.candidate_model_scores = model_scores
                        st.session_state.backtest_metrics = None
                        st.session_state.backtest_weights = None
                        st.session_state.backtest_note = ""
                        st.session_state.backtest_fold_df = None
                        st.session_state.candidate_df = None
                        st.session_state.candidate_groups = None
                        st.session_state.candidate_pool_df = None
                        st.session_state.position_7_report = None
                        st.session_state.danma_prediction = None
                        st.session_state.no_position_7_prediction = None
                        st.session_state.app_state.last_trained_at = now_str()
                        st.session_state.app_state.current_mode = "训练完成"
                        log_run_details("训练", time.perf_counter() - started_at, model_scores=model_scores, skipped=skipped)
                        st.rerun()
                    except Exception as exc:
                        st.error(f"训练失败：{exc}")
                        log_event(f"训练失败：{exc}")
            elif train_btn:
                st.warning("请先上传并校验数据。")

            if backtest_btn and clean_df is not None and feature_df is not None:
                backtest_spinner_text = (
                    "正在滚动回测，预计40-60秒..."
                    if st.session_state.backtest_mode == "滚动窗口"
                    else "正在回测：严格按时间顺序计算，预计40-60秒..."
                )
                with st.spinner(backtest_spinner_text):
                    try:
                        started_at = time.perf_counter()
                        fit_deep = bool(st.session_state.use_tabresnet)
                        if st.session_state.backtest_mode == "固定 80/20":
                            split_at = max(5, int(len(feature_df) * float(st.session_state.train_ratio)))
                            split_at = min(split_at, len(feature_df) - 1)
                            skipped = skipped_model_reasons(fit_deep, train_samples=split_at)
                            metrics, weights, note = run_fixed_backtest(
                                clean_df=clean_df,
                                feature_df=feature_df,
                                seed=int(st.session_state.seed),
                                train_ratio=float(st.session_state.train_ratio),
                                fit_deep=fit_deep,
                            )
                            fold_df = None
                        else:
                            skipped = skipped_model_reasons(
                                fit_deep,
                                train_samples=int(st.session_state.rolling_initial_train_size),
                            )
                            metrics, weights, note, fold_df = run_rolling_backtest(
                                clean_df=clean_df,
                                feature_df=feature_df,
                                seed=int(st.session_state.seed),
                                initial_train_size=int(st.session_state.rolling_initial_train_size),
                                batch_size=int(st.session_state.rolling_batch_size),
                                fit_deep=fit_deep,
                            )
                        st.session_state.backtest_metrics = metrics
                        st.session_state.backtest_weights = weights
                        st.session_state.backtest_note = note
                        st.session_state.backtest_fold_df = fold_df
                        st.session_state.candidate_df = None
                        st.session_state.candidate_groups = None
                        st.session_state.candidate_pool_df = None
                        st.session_state.position_7_report = None
                        st.session_state.danma_prediction = None
                        st.session_state.no_position_7_prediction = None
                        st.session_state.app_state.last_backtest_at = now_str()
                        st.session_state.app_state.current_mode = "回测完成"
                        verdict_text, _ = baseline_comparison_text(metrics)
                        log_run_details("回测", time.perf_counter() - started_at, metrics=metrics, skipped=skipped)
                        log_event(verdict_text)
                        st.rerun()
                    except Exception as exc:
                        st.error(f"回测失败：{exc}")
                        log_event(f"回测失败：{exc}")
                        metrics = st.session_state.backtest_metrics or {}
                        weights = st.session_state.backtest_weights or {}
                        note = "回测未完成。"
                        fold_df = None
            else:
                metrics = st.session_state.backtest_metrics or {}
                weights = st.session_state.backtest_weights or {}
                note = st.session_state.get("backtest_note", "") if metrics else "尚未运行时间顺序回测。"
                fold_df = st.session_state.get("backtest_fold_df")
            render_backtest_results(metrics, note, fold_df)

        if active_tab == "候选与胆码":
            st.markdown(
                f'<div class="fc3d-panel"><h3>候选与胆码工作区</h3><div class="fc3d-note">当前状态：{escape(app_stage())}。本页优先展示下一期3个胆码、三个位7码和候选Top20；所有结果只是模型排序。</div></div>',
                unsafe_allow_html=True,
            )
            if predict_btn and clean_df is not None and feature_df is not None:
                with st.spinner("正在生成候选，请稍候..."):
                    try:
                        started_at = time.perf_counter()
                        fit_deep = bool(st.session_state.use_tabresnet)
                        skipped = skipped_model_reasons(fit_deep, train_samples=len(feature_df))
                        trained_scores = st.session_state.trained_models
                        model_scores, ensemble, candidate_df, position_report, pool_df, candidate_groups = predict_full_scores(
                            clean_df=clean_df,
                            feature_df=feature_df,
                            seed=int(st.session_state.seed),
                            fit_deep=fit_deep,
                            backtest_weights=st.session_state.backtest_weights or None,
                            model_scores=trained_scores,
                        )
                        if trained_scores is None:
                            st.session_state.app_state.last_trained_at = now_str()
                        st.session_state.trained_models = model_scores
                        st.session_state.candidate_df = candidate_df
                        st.session_state.candidate_groups = candidate_groups
                        st.session_state.candidate_pool_df = pool_df
                        st.session_state.candidate_model_scores = model_scores
                        st.session_state.position_7_report = position_report
                        st.session_state.danma_prediction = build_danma_prediction(
                            clean_df=clean_df,
                            model_scores=model_scores,
                            ensemble_scores=ensemble,
                            data_report=report or {},
                            generated_at=now_str(),
                        )
                        st.session_state.no_position_7_prediction = build_no_position_7_prediction(
                            clean_df=clean_df,
                            model_scores=model_scores,
                            ensemble_scores=ensemble,
                            data_report=report or {},
                            generated_at=now_str(),
                        )
                        st.session_state.app_state.last_predict_at = now_str()
                        st.session_state.app_state.current_mode = "候选已生成"
                        log_run_details("候选生成", time.perf_counter() - started_at, model_scores=model_scores, skipped=skipped)
                        st.rerun()
                    except Exception as exc:
                        st.error(f"候选生成失败：{exc}")
                        log_event(f"候选生成失败：{exc}")
            if st.session_state.candidate_df is not None:
                render_candidate_table(st.session_state.candidate_df)
            else:
                render_candidate_placeholder()

        if active_tab == "导出":
            report_text = build_report_text(report, st.session_state.backtest_metrics, st.session_state.candidate_df)
            ensure_report_download(report_text)
            render_export_center(report_text)

    with right_col:
        render_right_rail()

    sync_current_mode()
    with topbar_slot.container():
        render_topbar(st.session_state.validation_report)

    st.caption(f"{SAFETY_TEXT}  {SIMULATION_TEXT}")


if __name__ == "__main__":
    main()

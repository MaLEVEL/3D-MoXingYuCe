import io
import sys
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import fc3d_ai_assistant as core
import fc3d_workbench as app


def make_history(rows: int = 160) -> pd.DataFrame:
    data = []
    start = pd.Timestamp("2025-01-01")
    for idx in range(rows):
        number = f"{idx % 10}{(idx // 3) % 10}{(idx // 7) % 10}"
        data.append(
            {
                "issue": f"2025{idx + 1:03d}",
                "date": (start + pd.Timedelta(days=idx)).strftime("%Y-%m-%d"),
                "number": number,
            }
        )
    return core.add_number_columns(pd.DataFrame(data))


def inspect_csv(text: str):
    raw = app.read_csv_bytes(text.encode("utf-8"))
    return app.inspect_and_clean(raw, "unit.csv", text.encode("utf-8"))


def test_default_history_configuration_uses_five_year_window():
    assert app.HISTORY_LOOKBACK_DAYS >= 365 * 5
    assert app.default_history_name() == "近5年开奖记录"
    assert app.BUILTIN_HISTORY_PATH.name == "fc3d_last_5_years_history.csv"


def test_normalize_history_rows_keeps_five_year_window_and_requires_stable_sample():
    rows = []
    start = pd.Timestamp("2020-01-01")
    for idx in range(2200):
        rows.append(
            {
                "issue": f"{2020000 + idx:07d}",
                "date": (start + pd.Timedelta(days=idx)).strftime("%Y-%m-%d"),
                "number": f"{idx % 1000:03d}",
            }
        )

    normalized = app.normalize_history_rows(rows, app.HISTORY_LOOKBACK_DAYS, "unit")
    latest = pd.to_datetime(normalized["date"]).max()
    earliest = pd.to_datetime(normalized["date"]).min()

    assert len(normalized) >= 1800
    assert (latest - earliest).days <= app.HISTORY_LOOKBACK_DAYS
    assert normalized["number"].str.fullmatch(r"\d{3}").all()


def test_cross_year_history_rows_are_sorted_by_date():
    start = pd.Timestamp("2023-12-01")
    rows = [
        {
            "issue": f"{2023000 + idx:07d}",
            "date": (start + pd.Timedelta(days=idx)).strftime("%Y-%m-%d"),
            "number": f"{idx % 1000:03d}",
        }
        for idx in range(1001)
    ]

    normalized = app.normalize_history_rows(list(reversed(rows)), app.HISTORY_LOOKBACK_DAYS, "unit")

    assert normalized["date"].is_monotonic_increasing
    assert normalized["date"].str[:4].nunique() >= 3
    assert normalized.iloc[0]["number"] == "000"


def test_update_history_repository_appends_missing_online_rows(tmp_path, monkeypatch):
    rows = []
    start = pd.Timestamp("2021-01-01")
    for idx in range(1005):
        rows.append(
            {
                "issue": f"{2021000 + idx:07d}",
                "date": (start + pd.Timedelta(days=idx)).strftime("%Y-%m-%d"),
                "number": f"{idx % 1000:03d}",
            }
        )
    online_df = pd.DataFrame(
        rows
        + [
            {"issue": "2023006", "date": "2023-10-03", "number": "006"},
            {"issue": "2023007", "date": "2023-10-04", "number": "007"},
        ]
    )
    history_path = tmp_path / "history.csv"
    pd.DataFrame(rows).to_csv(history_path, index=False, encoding="utf-8-sig")

    monkeypatch.setattr(app, "fetch_online_history", lambda days=app.HISTORY_LOOKBACK_DAYS: (online_df, "unit接口"))

    result = app.update_history_repository(history_path)
    saved = pd.read_csv(history_path, dtype="string", keep_default_na=False)

    assert result["added_count"] == 2
    assert result["source_name"] == "unit接口"
    assert result["previous_count"] == 1005
    assert result["row_count"] == 1007
    assert result["latest_issue"] == "2023007"
    assert saved["issue"].duplicated().sum() == 0
    assert saved.tail(2)["number"].tolist() == ["006", "007"]


def test_update_history_repository_does_not_duplicate_current_rows(tmp_path, monkeypatch):
    rows = []
    start = pd.Timestamp("2021-01-01")
    for idx in range(1005):
        rows.append(
            {
                "issue": f"{2021000 + idx:07d}",
                "date": (start + pd.Timedelta(days=idx)).strftime("%Y-%m-%d"),
                "number": f"{idx % 1000:03d}",
            }
        )
    online_df = pd.DataFrame(rows)
    history_path = tmp_path / "history.csv"
    online_df.to_csv(history_path, index=False, encoding="utf-8-sig")

    monkeypatch.setattr(app, "fetch_online_history", lambda days=app.HISTORY_LOOKBACK_DAYS: (online_df, "unit接口"))

    result = app.update_history_repository(history_path)
    saved = pd.read_csv(history_path, dtype="string", keep_default_na=False)

    assert result["added_count"] == 0
    assert result["row_count"] == 1005
    assert saved["issue"].nunique() == 1005


def test_workbench_no_longer_exposes_one_year_default_copy():
    source = (ROOT / "fc3d_workbench.py").read_text(encoding="utf-8")

    assert "近一年开奖记录" not in source
    assert "获取近一年开奖记录失败" not in source
    assert "点击开始回测后" not in source


def test_theme_tokens_include_high_contrast_light_and_dark_modes():
    assert app.THEME_OPTIONS == ("亮色", "暗色")
    assert app.normalize_theme_mode("未知") == "亮色"

    light_css = app.theme_css_variables("亮色")
    dark_css = app.theme_css_variables("暗色")

    assert "--text: #0B1220;" in light_css
    assert "--muted: #334155;" in light_css
    assert "--panel: #FFFFFF;" in light_css
    assert "--text: #F8FAFC;" in dark_css
    assert "--muted: #CBD5E1;" in dark_css
    assert "--panel: #151C2E;" in dark_css
    assert "--disabled-text: #475569;" in light_css
    assert "--disabled-text: #CBD5E1;" in dark_css


def test_normal_csv_loads_and_reports_hash():
    clean_df, report = inspect_csv("issue,date,number\n2025001,2025-01-01,123\n2025002,2025-01-02,456\n")

    assert len(clean_df) == 2
    assert report["kept_rows"] == 2
    assert report["date_min"] == "2025-01-01"
    assert report["date_max"] == "2025-01-02"
    assert report["data_version"]


def test_leading_zero_number_is_preserved():
    clean_df, report = inspect_csv("issue,date,number\n2025001,2025-01-01,007\n2025002,2025-01-02,018\n")

    assert clean_df["number"].tolist() == ["007", "018"]
    assert report["leading_zero_count"] == 2


def test_duplicate_issue_is_reported_and_cleaned():
    clean_df, report = inspect_csv(
        "issue,date,number\n"
        "2025001,2025-01-01,123\n"
        "2025001,2025-01-02,456\n"
        "2025002,2025-01-03,789\n"
    )

    assert report["duplicate_issue_count"] == 1
    assert len(report["duplicate_issue_rows"]) == 2
    assert clean_df["issue"].tolist() == ["2025001", "2025002"]


def test_invalid_number_is_reported_and_removed():
    clean_df, report = inspect_csv(
        "issue,date,number\n"
        "2025001,2025-01-01,123\n"
        "2025002,2025-01-02,12A\n"
        "2025003,2025-01-03,1000\n"
    )

    assert report["invalid_number_count"] == 2
    assert clean_df["number"].tolist() == ["123"]


def test_small_sample_warning_uses_1000_issue_threshold():
    _, report = inspect_csv("issue,date,number\n2025001,2025-01-01,123\n")

    assert report["warning_small_sample"] is True
    assert report["sample_warning_text"] == "当前样本量较少，模型结果仅适合学习和模拟研究。"


def test_core_backtest_warns_when_sample_is_under_1000(monkeypatch):
    df = make_history(80)
    feature_df = core.build_feature_table(df)
    monkeypatch.setattr(core, "fit_ml_models", lambda *args, **kwargs: {})

    output = io.StringIO()
    with redirect_stdout(output):
        core.run_backtest(df, feature_df, core.RunConfig(seed=42, train_ratio=0.8))

    assert "当前样本量较少，模型结果仅适合学习和模拟研究。" in output.getvalue()


def test_workbench_language_uses_candidate_sorting_terms():
    source = (ROOT / "fc3d_workbench.py").read_text(encoding="utf-8")

    for phrase in ["当前推荐7码", "最后预测", "所有预测结果"]:
        assert phrase not in source


def test_position_features_include_required_windows_and_no_future_data():
    df = make_history(150)
    feature_df = core.build_feature_table(df)
    first_target_row = feature_df.iloc[0]
    first_expected = core.build_history_feature(df.iloc[: int(first_target_row["row_index"])])

    for window in (30, 60, 120):
        assert f"roll{window}_bai_freq_0" in feature_df.columns
        assert f"roll{window}_shi_freq_0" in feature_df.columns
        assert f"roll{window}_ge_freq_0" in feature_df.columns

    for name in [
        "expdecay_bai_freq_0",
        "avg_omit_bai_0",
        "max_omit_bai_0",
        "heat_bai_0",
        "prev_sum3",
        "prev_span",
        "prev_is_zusan",
        "prev_is_zuliu",
    ]:
        assert name in feature_df.columns
        assert first_target_row[name] == first_expected[name]

    removed_feature_names = [
        "prev_odd_count",
        "prev_big_count",
        "roll30_odd_mean",
        "roll30_big_mean",
        "roll30_same_pos_repeat_mean",
        "roll30_overlap_repeat_mean",
    ]
    for name in removed_feature_names:
        assert name not in feature_df.columns


def test_fixed_backtest_stat_models_use_only_prior_history(monkeypatch):
    df = make_history(80)
    feature_df = core.build_feature_table(df)
    seen_history_lengths: list[int] = []

    def fake_stat_scores(history: pd.DataFrame, seed: int):
        seen_history_lengths.append(len(history))
        return {
            "history_frequency": np.ones(1000, dtype=float) / 1000,
            "random_baseline": np.ones(1000, dtype=float) / 1000,
        }

    monkeypatch.setattr(app, "get_stat_model_scores", fake_stat_scores)
    monkeypatch.setattr(app, "build_ml_factories", lambda seed: {})

    app.run_fixed_backtest(df, feature_df, seed=42, train_ratio=0.8, fit_deep=False)

    split_at = max(5, int(len(feature_df) * 0.8))
    expected_row_indices = feature_df.iloc[split_at:]["row_index"].astype(int).tolist()
    assert seen_history_lengths == expected_row_indices


def test_rolling_backtest_uses_initial_window_batches_and_prior_history(monkeypatch):
    df = make_history(1105)
    feature_df = core.build_feature_table(df)
    seen_history_lengths: list[int] = []

    def fake_stat_scores(history: pd.DataFrame, seed: int):
        seen_history_lengths.append(len(history))
        return {
            "history_frequency": np.ones(1000, dtype=float) / 1000,
            "random_baseline": np.ones(1000, dtype=float) / 1000,
        }

    monkeypatch.setattr(app, "get_stat_model_scores", fake_stat_scores)
    monkeypatch.setattr(app, "build_ml_factories", lambda seed: {})

    metrics, weights, note, fold_df = app.run_rolling_backtest(
        df,
        feature_df,
        seed=42,
        initial_train_size=1000,
        batch_size=25,
        fit_deep=False,
    )

    assert metrics["ensemble_equal"]["samples"] == len(feature_df) - 1000
    assert fold_df.iloc[0]["train_samples"] == 1000
    assert fold_df["test_samples"].sum() == len(feature_df) - 1000
    assert fold_df["fold"].nunique() == int(np.ceil((len(feature_df) - 1000) / 25))
    assert all(fold_df["train_end_row"].astype(int) < fold_df["test_start_row"].astype(int))
    assert seen_history_lengths == feature_df.iloc[1000:]["row_index"].astype(int).tolist()
    assert "回测模式：滚动回测" in note
    assert "滚动窗口数量" in note
    assert abs(sum(weights.values()) - 1.0) < 1e-9


def test_rolling_backtest_rejects_too_small_initial_window():
    df = make_history(120)
    feature_df = core.build_feature_table(df)

    try:
        app.run_rolling_backtest(df, feature_df, seed=42, initial_train_size=1000, batch_size=20, fit_deep=False)
    except ValueError as exc:
        assert "初始训练窗口" in str(exc)
    else:
        raise AssertionError("rolling backtest should reject data shorter than initial window")


def test_three_position_top7_coverage_calculation():
    strong = np.array([0.12] * 7 + [0.01] * 3)
    strong = strong / strong.sum()
    weak = strong[::-1] / strong[::-1].sum()
    scores_hit = core.code_scores_from_position_probs((strong, strong, strong))
    scores_miss = core.code_scores_from_position_probs((strong, strong, strong))

    metrics = core.evaluate_score_rows(["000", "999"], [scores_hit, scores_miss])

    assert metrics["position_top7_all_hit_rate"] == 0.5


def test_backtest_baseline_verdict_uses_required_separate_sentences():
    metrics = {
        "ensemble_equal": {
            "samples": 1200,
            "position_top7_all_hit_rate": 0.32,
            "top10_hit_rate": 0.02,
        }
    }

    text, _ = app.baseline_comparison_text(metrics)

    assert "三位7码未超过随机基线，暂未证明有效。" in text
    assert "直选Top10本次高于随机基线，但不能证明长期有效。" in text


def test_direct_random_baselines_are_available_for_top10_top20_top50():
    table = app.metrics_table({"random_baseline": {"samples": 1000}})
    row = table.iloc[0]

    assert row["Top10"] == app.THEORETICAL_TOP10_BASELINE
    assert row["Top20"] == app.THEORETICAL_TOP20_BASELINE
    assert row["Top50"] == app.THEORETICAL_TOP50_BASELINE


def test_direct_backtest_table_shows_rates_hits_and_random_baselines():
    metrics = {
        "ensemble_equal": {
            "samples": 100,
            "top10_hit_rate": 0.02,
            "top20_hit_rate": 0.03,
            "top50_hit_rate": 0.06,
        }
    }

    table = app.direct_backtest_table(metrics)
    row = table.iloc[0]

    assert row["Top10命中率"] == "2.000%"
    assert row["Top10命中次数"] == "2/100"
    assert row["Top10随机基线"] == "1.000%"
    assert row["Top20命中次数"] == "3/100"
    assert row["Top50随机基线"] == "5.000%"


def test_direct_metric_table_can_split_top10_top20_top50():
    metrics = {
        "ensemble_equal": {
            "samples": 100,
            "top10_hit_rate": 0.02,
            "top20_hit_rate": 0.03,
            "top50_hit_rate": 0.06,
        }
    }

    table = app.direct_metric_table(metrics, "Top50")

    assert list(table.columns) == ["模型", "样本", "Top50命中率", "Top50命中次数", "Top50随机基线"]
    assert table.iloc[0]["Top50命中次数"] == "6/100"
    assert table.iloc[0]["Top50随机基线"] == "5.000%"


def test_default_backtest_mode_prefers_rolling_window():
    source = (ROOT / "fc3d_workbench.py").read_text(encoding="utf-8")

    assert '["固定 80/20", "滚动窗口"],\n            index=1' in source


def test_baseline_summary_is_split_into_short_rows_and_statistical_conclusion():
    metrics = {
        "ensemble_equal": {
            "samples": 755,
            "position_top7_all_hit_rate": 0.345695,
            "top10_hit_rate": 0.005298,
            "top20_hit_rate": 0.01457,
            "top50_hit_rate": 0.042384,
        }
    }

    summary = app.baseline_summary_dataframe(metrics)
    text = "\n".join(summary["结论"].astype(str).tolist())

    assert summary["项目"].tolist() == ["三位7码", "直选Top10", "综合判断", "统计结论"]
    assert "略高于基线，但不显著" in text
    assert "未超过基线" in text
    assert "暂未证明长期有效" in text
    assert "未达到显著优势" in text


def test_css_replaces_expander_icon_text_and_upload_copy():
    source = (ROOT / "fc3d_workbench.py").read_text(encoding="utf-8")

    assert "上传历史开奖数据CSV（可选）" in source
    assert "把历史开奖CSV拖到这里" in source
    assert "没有CSV也可以直接点击下方按钮加载近5年开奖记录" in source
    assert "选择CSV文件" in source
    assert "字段必须为 `issue,date,number`" not in source
    assert "stExpanderToggleIcon" in source


def test_streamlit_internal_upload_and_expander_text_are_hidden_by_real_selectors():
    source = (ROOT / "fc3d_workbench.py").read_text(encoding="utf-8")

    assert '[data-testid="stFileUploaderDropzoneInstructions"] > *' in source
    assert '[data-testid="stFileUploaderDropzone"] button *' in source
    assert '[data-testid="stIconMaterial"]' in source
    assert 'content: ""' in source
    assert "border-left:" in source
    assert '[data-testid="stFileUploaderDropzone"] button[data-testid="stBaseButton-secondary"]' not in source


def test_expander_icon_uses_css_shape_not_visible_ligature_text():
    source = (ROOT / "fc3d_workbench.py").read_text(encoding="utf-8")

    assert '[data-testid="stExpander"] summary::before' in source
    assert "border-left:" in source
    assert '[data-testid="stExpander"] summary [data-testid="stIconMaterial"]' in source
    assert "keyboard_arrow_right" not in source.split("<style>", 1)[-1].split("</style>", 1)[0]


def test_desktop_sidebar_expand_button_is_visible_outside_mobile_media_query():
    source = (ROOT / "fc3d_workbench.py").read_text(encoding="utf-8")
    desktop_css = source.split("@media (max-width: 700px)", 1)[0]

    assert '[data-testid="stExpandSidebarButton"]' in desktop_css
    assert "visibility: visible !important" in desktop_css
    assert "position: fixed !important" in desktop_css
    assert "pointer-events: auto !important" in desktop_css


def test_direct_tables_include_scroll_hint_and_compact_container():
    source = (ROOT / "fc3d_workbench.py").read_text(encoding="utf-8")

    assert "fc3d-direct-table-wrap" in source
    assert "表格可横向滚动" in source


def test_rolling_backtest_spinner_mentions_waiting():
    source = (ROOT / "fc3d_workbench.py").read_text(encoding="utf-8")

    assert "正在滚动回测，预计40-60秒" in source


def test_backtest_audit_table_adds_hits_expected_diff_and_interval():
    metrics = {
        "ensemble_equal": {
            "samples": 100,
            "bai_top7_hit_rate": 0.72,
            "shi_top7_hit_rate": 0.68,
            "ge_top7_hit_rate": 0.71,
            "position_top7_all_hit_rate": 0.35,
            "top10_hit_rate": 0.02,
            "top20_hit_rate": 0.03,
            "top50_hit_rate": 0.06,
        }
    }

    table = app.backtest_audit_table(metrics)

    assert len(table) == 7
    assert {"指标", "命中次数", "理论期望", "差值", "95%置信区间", "随机基线", "显著性提示"}.issubset(table.columns)
    position_row = table.loc[table["指标"] == "三位同时Top7"].iloc[0]
    assert position_row["命中次数"] == "35/100"
    assert position_row["理论期望"] == "34.3"
    assert position_row["差值"] == "+0.7"
    assert "[" in position_row["95%置信区间"]
    assert "近似p值" in table.columns
    assert "是否显著" in table.columns
    assert "小样本" in position_row["显著性提示"]


def test_backtest_audit_heading_names_p_value_and_significance():
    source = (ROOT / "fc3d_workbench.py").read_text(encoding="utf-8")

    assert "统计审计：命中次数 / 理论期望 / 差值 / 95%置信区间 / 近似p值 / 是否显著" in source
    assert "统计审计：命中次数 / 理论期望 / 差值 / 95%置信区间 / 近似二项检验" not in source


def test_343_pool_count_and_position_report():
    df = make_history(80)
    ensemble = np.ones(1000, dtype=float) / 1000
    report = app.build_position_7_code_report(ensemble)
    pool_df = app.build_position_pool_dataframe(df, {"history_frequency": ensemble, "random_baseline": ensemble[::-1]}, ensemble, report)

    assert report["pool_size"] == 343
    assert len(pool_df) == 343
    assert all(len(report["positions"][key]["digits"]) == 7 for key in ["bai", "shi", "ge"])
    assert all(len(report["positions"][key]["excluded"]) == 3 for key in ["bai", "shi", "ge"])


def test_position_7_report_includes_high_confidence_threshold():
    scores = np.arange(1, 1001, dtype=float)
    report = app.build_position_7_code_report(scores, app.POSITION_7_CONFIDENCE_RULE)

    threshold = report["confidence_threshold"]

    assert threshold["model"] == "history_frequency"
    assert threshold["feature"] == "min_top7_share"
    assert threshold["direction"] == "<="
    assert threshold["target_rate"] == 0.40
    assert threshold["validation_hits"] == 23
    assert threshold["validation_selected"] == 57
    assert "passes" in threshold


def test_candidate_score_display_is_not_probability_text():
    df = pd.DataFrame(
        [{"rank": 1, "number": "007", "score": 0.012345, "source_models": "历史频率", "reason": "测试"}]
    )

    formatted = app.format_candidate_group(df)

    assert formatted.loc[0, "模型相对分数"] == "0.0123"
    assert "%" not in formatted.loc[0, "模型相对分数"]


def test_candidate_model_context_distinguishes_models_and_weight_source():
    model_scores = {
        "history_frequency": np.ones(1000, dtype=float) / 1000,
        "random_forest": np.ones(1000, dtype=float) / 1000,
        "random_baseline": np.ones(1000, dtype=float) / 1000,
    }
    metrics = {
        "history_frequency": {"samples": 1200, "position_top7_all_hit_rate": 0.36, "top10_hit_rate": 0.011},
        "random_forest": {"samples": 1200, "position_top7_all_hit_rate": 0.34, "top10_hit_rate": 0.018},
    }
    weights = {"history_frequency": 0.4, "random_forest": 0.6}

    context = app.candidate_model_context_dataframe(model_scores, metrics, weights)
    joined = "\n".join(context["内容"].astype(str).tolist())

    assert context["项目"].tolist() == ["当前候选使用模型", "回测最佳模型", "最终权重来源"]
    assert "历史频率" in joined
    assert "随机森林" in joined
    assert "回测权重" in joined


def test_compact_candidate_grid_html_uses_dense_grid_class():
    df = pd.DataFrame(
        [
            {"rank": 1, "number": "007", "score": 0.0123, "source_models": "历史频率", "reason": "测试"},
            {"rank": 2, "number": "018", "score": 0.0111, "source_models": "随机森林", "reason": "测试"},
        ]
    )

    html = app.compact_candidate_grid_html(df)

    assert "fc3d-compact-candidate-grid" in html
    assert "007" in html
    assert "模型相对分数" in html


def test_long_running_status_copy_mentions_steps_and_elapsed_time():
    source = (ROOT / "fc3d_workbench.py").read_text(encoding="utf-8")

    for phrase in ["正在校验数据", "正在训练", "正在回测", "正在生成候选", "耗时"]:
        assert phrase in source


def test_productized_action_area_uses_compact_next_step_and_specific_button_copy():
    source = (ROOT / "fc3d_workbench.py").read_text(encoding="utf-8")

    assert "fc3d-action-grid" in source
    assert "fc3d-disabled-reason" in source
    assert "下一步" in source
    for phrase in [
        "加载近5年开奖记录",
        "校验数据",
        "训练模型",
        "开始滚动回测",
        "生成候选与胆码",
        "导出分析报告",
    ]:
        assert phrase in source
    assert "重新加载近5年开奖记录会清空当前训练、回测和候选结果" in source


def test_workspace_navigation_uses_four_short_work_areas():
    source = (ROOT / "fc3d_workbench.py").read_text(encoding="utf-8")

    assert '["数据", "回测", "候选与胆码", "导出"]' in source
    assert 'st.session_state.active_tab = "候选与胆码"' in source
    assert 'if active_tab == "数据":' in source
    assert 'if active_tab == "候选与胆码":' in source


def test_backtest_result_area_has_conclusion_card_and_significance_badges():
    source = (ROOT / "fc3d_workbench.py").read_text(encoding="utf-8")

    assert "回测结论卡片" in source
    assert "fc3d-backtest-conclusion-grid" in source
    assert "当前模型暂未证明长期有效，仅可作为模拟排序参考。" in source
    assert "fc3d-sig-yes" in source
    assert "fc3d-sig-no" in source


def test_candidate_and_export_pages_prioritize_user_facing_results():
    source = (ROOT / "fc3d_workbench.py").read_text(encoding="utf-8")

    assert "fc3d-danma-card" in source
    assert "胆码只是模型排序结果，不是中奖概率，不代表真实预测。" in source
    assert "主候选 Top20" in source
    assert "分散候选 Top20" in source
    assert "冷门参考 Top20" in source
    assert "报告包含内容" in source
    assert "导出完整分析报告" in source
    assert "导出胆码与候选CSV" in source


def test_candidate_and_danma_export_dataframe_uses_current_display_results():
    ranking = pd.DataFrame(
        [
            {"rank": 1, "digit": 6, "score": 0.19, "source_models": "历史频率"},
            {"rank": 2, "digit": 4, "score": 0.18, "source_models": "遗漏值"},
            {"rank": 3, "digit": 9, "score": 0.17, "source_models": "和值分布"},
        ]
    )
    main_df = pd.DataFrame(
        [
            {"rank": 1, "number": "649", "score": 0.012345, "source_models": "历史频率", "reason": "主候选"},
        ]
    )
    diverse_df = pd.DataFrame(
        [
            {"rank": 8, "number": "027", "score": 0.009, "source_models": "随机森林", "reason": "分散"},
        ]
    )
    long_tail_df = pd.DataFrame(
        [
            {"rank": 300, "number": "135", "score": 0.001, "source_models": "贝叶斯平滑", "reason": "冷门"},
        ]
    )
    app.st.session_state.danma_prediction = {"ranking": ranking}
    app.st.session_state.candidate_df = main_df
    app.st.session_state.candidate_groups = {
        "main": main_df,
        "diverse": diverse_df,
        "long_tail": long_tail_df,
    }

    export_df = app.candidate_and_danma_export_dataframe()

    assert export_df["类型"].tolist() == [
        "下一期3个胆码",
        "下一期3个胆码",
        "下一期3个胆码",
        "主候选Top20",
        "分散候选Top20",
        "冷门参考Top20",
    ]
    assert export_df.loc[0, "号码"] == "6"
    assert export_df.loc[3, "号码"] == "649"
    assert export_df.loc[3, "模型相对分数"] == "0.012345"
    assert "中奖概率" not in ",".join(export_df.columns)


def test_export_343_pool_uses_relative_score_fields():
    df = make_history(80)
    ensemble = core.normalize_scores(np.arange(1, 1001, dtype=float))
    report = app.build_position_7_code_report(ensemble)
    pool_df = app.build_position_pool_dataframe(df, {"history_frequency": ensemble}, ensemble, report)

    export_pool = app.format_343_export_dataframe(pool_df)

    assert len(export_pool) == 343
    assert list(export_pool.columns) == ["rank", "number", "model_relative_score", "source_models", "reason"]
    assert "probability" not in ",".join(export_pool.columns).lower()


def test_model_score_calibration_softens_spiky_model_scores():
    spiky = np.zeros(1000, dtype=float)
    spiky[0] = 1.0

    calibrated = core.calibrate_model_scores(spiky)

    assert calibrated[0] < 1.0
    assert calibrated[0] > calibrated[1]
    np.testing.assert_allclose(calibrated.sum(), 1.0)


def test_random_baseline_does_not_participate_in_ensemble_scoring():
    model_score = core.normalize_scores(np.arange(1, 1001, dtype=float))
    random_score = model_score[::-1]

    without_baseline = core.ensemble_scores({"history_frequency": model_score})
    with_baseline = core.ensemble_scores({"history_frequency": model_score, "random_baseline": random_score})

    np.testing.assert_allclose(with_baseline, without_baseline)


def test_random_baseline_only_returns_neutral_scores():
    random_score = core.normalize_scores(np.arange(1, 1001, dtype=float))

    combined = core.ensemble_scores({"random_baseline": random_score})

    np.testing.assert_allclose(combined, np.ones(1000, dtype=float) / 1000)


def test_diverse_and_long_tail_candidates_are_not_empty():
    df = make_history(80)
    ensemble = core.normalize_scores(np.arange(1, 1001, dtype=float))
    report = app.build_position_7_code_report(ensemble)
    pool_df = app.build_position_pool_dataframe(df, {"history_frequency": ensemble}, ensemble, report)
    groups = app.build_candidate_groups(pool_df, top_k=20)

    assert not groups["main"].empty
    assert not groups["diverse"].empty
    assert not groups["long_tail"].empty
    assert len(groups["main"]) == 20
    assert len(groups["diverse"]) == 20
    assert len(groups["long_tail"]) == 20


def test_danma_prediction_builds_three_unique_digits_and_full_ranking():
    df = make_history(160)
    ensemble = core.normalize_scores(np.arange(1, 1001, dtype=float))
    model_scores = {
        "history_frequency": ensemble,
        "random_forest": ensemble[::-1],
    }

    prediction = app.build_danma_prediction(
        clean_df=df,
        model_scores=model_scores,
        ensemble_scores=ensemble,
        data_report={"data_version": "unit-v1", "date_max": "2025-06-01"},
        generated_at="2026-06-05 10:00:00",
    )

    assert len(prediction["digits"]) == 3
    assert len(set(prediction["digits"])) == 3
    assert all(0 <= digit <= 9 for digit in prediction["digits"])
    assert prediction["data_version"] == "unit-v1"
    assert prediction["latest_issue"] == df.iloc[-1]["issue"]
    assert prediction["latest_date"] == "2025-06-01"

    ranking = prediction["ranking"]
    assert len(ranking) == 10
    assert ranking["rank"].tolist() == list(range(1, 11))
    assert ranking["score"].is_monotonic_decreasing
    assert ranking.head(3)["digit"].tolist() == prediction["digits"]
    assert {"position_marginal_score", "unique_presence_score", "source_models"}.issubset(ranking.columns)
    assert "模型相对分数不是中奖概率，只用于排序" in prediction["score_note"]


def test_danma_score_components_are_normalized_before_weighting():
    df = make_history(160)
    ensemble = core.normalize_scores(np.arange(1, 1001, dtype=float) ** 3)
    prediction = app.build_danma_prediction(
        clean_df=df,
        model_scores={"history_frequency": ensemble},
        ensemble_scores=ensemble,
        data_report={},
        generated_at="2026-06-05 10:00:00",
    )

    ranking = prediction["ranking"]
    for column in app.DANMA_COMPONENT_COLUMNS:
        assert ranking[column].between(0.0, 1.0).all(), column
    assert ranking["score"].between(0.0, 1.0).all()
    assert app.DANMA_SCORE_WEIGHTS == {
        "combination_marginal_score": 0.3429,
        "near30_frequency": 0.0,
        "near100_frequency": 0.0,
        "position_frequency": 0.0,
        "omission_rebound": 0.1322,
        "sum_distribution_support": 0.2880,
        "heat_cold_stability": 0.2370,
    }


def test_evaluate_danma_predictions_counts_required_metrics():
    metrics = app.evaluate_danma_predictions(
        actual_numbers=["123", "666", "789"],
        danma_rows=[[1, 4, 5], [6, 7, 8], [0, 1, 2]],
    )

    assert metrics["danma_samples"] == 3
    assert metrics["danma_at_least_1_position_hit_rate"] == 2 / 3
    assert metrics["danma_at_least_2_position_hit_rate"] == 1 / 3
    assert metrics["danma_all_3_positions_in_set_rate"] == 1 / 3
    assert metrics["danma_all_3_danma_in_unique_draw_rate"] == 0.0
    assert metrics["danma_average_position_hit_count"] == 4 / 3


def test_evaluate_no_position_7_predictions_counts_23_metric():
    metrics = app.evaluate_no_position_7_predictions(
        actual_numbers=["123", "666", "789", "012"],
        digit_rows=[
            [1, 2, 3, 4, 5, 6, 7],
            [0, 1, 2, 3, 4, 5, 6],
            [0, 1, 2, 3, 4, 5, 6],
            [3, 4, 5, 6, 7, 8, 9],
        ],
    )

    assert metrics["no_position_7_samples"] == 4
    assert metrics["no_position_7_at_least_1_position_hit_rate"] == 0.5
    assert metrics["no_position_7_at_least_2_position_hit_rate"] == 0.5
    assert metrics["no_position_7_all_3_positions_in_set_rate"] == 0.5
    assert metrics["no_position_7_average_position_hit_count"] == 1.5


def test_no_position_7_audit_table_includes_23_baseline_and_target():
    metrics = {
        "ensemble_equal": {
            "no_position_7_samples": 1000,
            "no_position_7_at_least_1_position_hit_rate": 0.974,
            "no_position_7_at_least_2_position_hit_rate": 0.821,
            "no_position_7_all_3_positions_in_set_rate": 0.344,
            "no_position_7_average_position_hit_count": 2.11,
        }
    }

    table = app.no_position_7_backtest_audit_table(metrics)
    summary = app.no_position_7_backtest_summary_table(metrics)

    row_23 = table.loc[table["指标"] == "7码=23"].iloc[0]
    assert row_23["随机基线"] == "78.400%"
    assert row_23["目标82%"] == "达到"
    assert row_23["命中次数"] == "821/1000"
    assert summary.loc[summary["项目"] == "7码=23准确率", "值"].iloc[0] == "82.100%"


def test_no_position_7_prediction_includes_separate_confidence_threshold():
    df = make_history(160)
    ensemble = core.normalize_scores(np.arange(1, 1001, dtype=float))
    prediction = app.build_no_position_7_prediction(
        clean_df=df,
        model_scores={"sum_distribution": ensemble, "ensemble_equal": ensemble[::-1]},
        ensemble_scores=ensemble,
        data_report={"data_version": "unit-v1", "date_max": "2025-06-01"},
        generated_at="2026-06-05 10:00:00",
    )

    threshold = prediction["confidence_threshold"]

    assert threshold["model"] == app.NO_POSITION_7_CONFIDENCE_RULE["model"]
    assert threshold["feature"] == "top_sum"
    assert threshold["direction"] == ">="
    assert threshold["conditions"] == []
    assert threshold["value"] > 0.0
    assert "passes" in threshold
    assert app.NO_POSITION_7_CONFIDENCE_RULE["weights"] != app.DANMA_CONFIDENCE_RULE["weights"]


def test_quick_scheme_table_includes_current_7_and_danma_thresholds():
    table = app.quick_scheme_dataframe()
    names = table["方案"].tolist()

    assert len(table) == 4
    assert "不定位7码高置信=23" in names
    assert "定位7码高置信全中" in names
    assert "三胆码最优独立方案" in names
    assert "直选20注全覆盖方案" in names

    seven_row = table.loc[table["方案"] == "不定位7码高置信=23"].iloc[0]
    position_row = table.loc[table["方案"] == "定位7码高置信全中"].iloc[0]
    danma_row = table.loc[table["方案"] == "三胆码最优独立方案"].iloc[0]
    direct_row = table.loc[table["方案"] == "直选20注全覆盖方案"].iloc[0]

    assert "c=0.1,o=0.2,s=0.4,h=0.3" in seven_row["权重/阈值"]
    assert "top_sum>=4.000" in seven_row["权重/阈值"]
    assert seven_row["近一年验证"] == "7码=23 82.07%，覆盖329/351"
    assert "history_frequency" in position_row["权重/阈值"]
    assert "min_top7_share<=0.717" in position_row["权重/阈值"]
    assert position_row["近一年验证"] == "三位全中 40.35%，覆盖57/173"
    assert "c=0.3429,o=0.1322,s=0.288,h=0.237" in danma_row["权重/阈值"]
    assert danma_row["近一年验证"] == "至少命中1个 68.09%，覆盖351/351"
    assert direct_row["权重/阈值"] == "h=0,b=0.3,o=0.2,s=0.5,p=0；全量输出"
    assert direct_row["近一年验证"] == "近半年Top20 4.62%，覆盖173/173"


def test_quick_prediction_scheme_buttons_use_separate_configs():
    app.apply_quick_prediction_scheme("no_position_7_coverage")
    app.apply_quick_prediction_scheme("position_7_high_confidence")
    app.apply_quick_prediction_scheme("danma_optimal")
    app.apply_quick_prediction_scheme("direct_top20_full_coverage")

    seven_rule = app.get_active_no_position_7_rule()
    position_rule = app.get_active_position_7_rule()
    danma_rule = app.get_active_danma_rule()
    direct_weights = app.st.session_state.backtest_weights

    assert seven_rule["model"] == "sum_distribution"
    assert seven_rule["threshold"] == 4.000389
    assert seven_rule["weights"]["sum_distribution_support"] == 0.4
    assert position_rule["model"] == "history_frequency"
    assert position_rule["feature"] == "min_top7_share"
    assert position_rule["threshold"] == 0.716771789
    assert danma_rule["model"] == "ensemble_equal"
    assert danma_rule["threshold"] is None
    assert danma_rule["weights"]["sum_distribution_support"] == 0.2880
    assert seven_rule["weights"] != danma_rule["weights"]
    assert direct_weights == {
        "history_frequency": 0.0,
        "bayes_smooth_frequency": 0.3,
        "omission": 0.2,
        "sum_distribution": 0.5,
        "position_frequency": 0.0,
    }


def test_danma_backtest_audit_table_includes_baseline_and_conclusion():
    metrics = {
        "ensemble_equal": {
            "danma_samples": 1000,
            "danma_at_least_1_position_hit_rate": 0.66,
            "danma_at_least_2_position_hit_rate": 0.217,
            "danma_all_3_positions_in_set_rate": 0.027,
            "danma_all_3_danma_in_unique_draw_rate": 0.006,
            "danma_average_position_hit_count": 0.9,
        }
    }

    table = app.danma_backtest_audit_table(metrics)
    conclusion = app.danma_statistical_conclusion(metrics)

    assert len(table) == 5
    assert {"指标", "实际命中率", "命中次数", "随机基线", "差值", "95%置信区间", "近似p值", "是否显著", "结论"}.issubset(table.columns)
    assert table.iloc[0]["随机基线"].endswith("%")
    average_row = table.loc[table["指标"] == "平均位置命中数"].iloc[0]
    assert average_row["实际命中率"] == "0.900"
    assert average_row["随机基线"] == "0.900"
    assert average_row["命中次数"] == "900/3000"
    assert conclusion in {"统计结论：未达到显著优势", "统计结论：达到显著优势"}


def test_rolling_backtest_adds_danma_metrics_without_future_data(monkeypatch):
    df = make_history(1105)
    feature_df = core.build_feature_table(df)
    seen_history_lengths: list[int] = []
    seen_danma_history_lengths: list[int] = []
    original_history_components = app.build_danma_history_components

    def fake_stat_scores(history: pd.DataFrame, seed: int):
        seen_history_lengths.append(len(history))
        return {
            "history_frequency": np.ones(1000, dtype=float) / 1000,
            "random_baseline": np.ones(1000, dtype=float) / 1000,
        }

    def tracking_history_components(history: pd.DataFrame):
        seen_danma_history_lengths.append(len(history))
        return original_history_components(history)

    monkeypatch.setattr(app, "get_stat_model_scores", fake_stat_scores)
    monkeypatch.setattr(app, "build_ml_factories", lambda seed: {})
    monkeypatch.setattr(app, "build_danma_history_components", tracking_history_components)

    metrics, weights, note, fold_df = app.run_rolling_backtest(
        df,
        feature_df,
        seed=42,
        initial_train_size=1000,
        batch_size=25,
        fit_deep=False,
    )

    expected_row_indices = feature_df.iloc[1000:]["row_index"].astype(int).tolist()
    assert metrics["ensemble_equal"]["danma_samples"] == len(feature_df) - 1000
    assert "danma_at_least_1_position_hit_rate" in metrics["ensemble_equal"]
    assert "danma_all_3_danma_in_unique_draw_rate" in metrics["ensemble_equal"]
    assert metrics["ensemble_equal"]["no_position_7_samples"] == len(feature_df) - 1000
    assert "no_position_7_at_least_2_position_hit_rate" in metrics["ensemble_equal"]
    assert "no_position_7_all_3_positions_in_set_rate" in metrics["ensemble_equal"]
    assert seen_history_lengths == expected_row_indices
    assert sorted(set(seen_danma_history_lengths)) == expected_row_indices
    assert not fold_df.empty
    assert weights


def test_report_text_includes_danma_prediction_ranking_and_audit():
    ranking = pd.DataFrame(
        [
            {"rank": rank, "digit": digit, "score": 0.2 - rank * 0.01, "source_models": "历史频率 / 遗漏值"}
            for rank, digit in enumerate([6, 4, 9, 0, 2, 3, 5, 7, 8, 1], start=1)
        ]
    )
    app.st.session_state.danma_prediction = {
        "digits": [6, 4, 9],
        "ranking": ranking,
        "generated_at": "2026-06-05 10:00:00",
        "data_version": "unit-v1",
        "latest_issue": "2025123",
        "latest_date": "2025-06-01",
        "score_note": "模型相对分数不是中奖概率，只用于排序。",
    }
    app.st.session_state.no_position_7_prediction = {
        "digits": [6, 4, 9, 0, 2, 3, 5],
        "ranking": ranking,
        "generated_at": "2026-06-05 10:00:00",
        "data_version": "unit-v1",
        "latest_issue": "2025123",
        "latest_date": "2025-06-01",
        "score_note": "模型相对分数不是中奖概率，只用于排序。",
    }
    metrics = {
        "ensemble_equal": {
            "danma_samples": 1000,
            "danma_at_least_1_position_hit_rate": 0.657,
            "danma_at_least_2_position_hit_rate": 0.216,
            "danma_all_3_positions_in_set_rate": 0.027,
            "danma_all_3_danma_in_unique_draw_rate": 0.006,
            "danma_average_position_hit_count": 0.9,
            "no_position_7_samples": 1000,
            "no_position_7_at_least_1_position_hit_rate": 0.973,
            "no_position_7_at_least_2_position_hit_rate": 0.784,
            "no_position_7_all_3_positions_in_set_rate": 0.343,
            "no_position_7_average_position_hit_count": 2.1,
        }
    }

    text = app.build_report_text(
        report={"file_name": "unit.csv", "data_version": "unit-v1"},
        backtest_metrics=metrics,
        candidate_df=pd.DataFrame(),
    )

    assert "下一期3个胆码" in text
    assert "胆码：6、4、9" in text
    assert "0-9完整分数排名" in text
    assert "胆码回测统计审计" in text
    assert "不定位7码" in text
    assert "7码=23准确率" in text
    assert "不定位7码回测统计审计" in text
    assert "当前胆码模型暂未证明长期有效，仅可作为模拟排序参考。" in text
    assert app.SAFETY_TEXT in text


def test_workbench_forbidden_safety_copy_is_absent():
    source = (ROOT / "fc3d_workbench.py").read_text(encoding="utf-8")

    phrases = [
        "必" + "中",
        "稳" + "赢",
        "精准" + "预测",
        "完美" + "预测",
        "保证" + "命中",
    ]
    for phrase in phrases:
        assert phrase not in source
    assert app.SAFETY_TEXT in source


def test_confidence_threshold_supports_and_conditions():
    ranking = pd.DataFrame(
        {
            "rank": list(range(1, 11)),
            "digit": list(range(10)),
            "score": [1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.2, 0.15, 0.1],
        }
    )
    rule = {
        "model": "sum_distribution",
        "conditions": [
            {"feature": "top_sum", "direction": ">=", "threshold": 5.0},
            {"feature": "bottom_sum", "direction": ">=", "threshold": 0.6},
        ],
    }

    status = app.build_confidence_threshold_status(ranking, rule, pool_size=7)

    assert status["passes"] is False
    assert status["conditions"][0]["passes"] is True
    assert status["conditions"][1]["passes"] is False

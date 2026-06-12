import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from pathlib import Path

from core.config import (
    format_station_label,
    station_key_from_display,
    line_key_from_station,
    find_station_by_name,
    select_line_station,
)
from core.predictor import predict, predict_day, get_congestion, get_station_avg

BASE_DIR = Path(__file__).resolve().parents[1]
DATASET_PATH = BASE_DIR / "data" / "processed" / "final_dataset_line1_8_230101-241231.csv"


@st.cache_data(show_spinner=False)
def load_metric_sample(station_key, sample_size):
    if not DATASET_PATH.exists():
        return pd.DataFrame()

    station_name, line_name = station_key.rsplit("_", 1)
    df = pd.read_csv(
        DATASET_PATH,
        usecols=["날짜", "역명", "호선", "시간", "승차인원", "하차인원", "기온", "강수량", "적설"],
        parse_dates=["날짜"],
    )
    matched = df[(df["역명"].astype(str) == station_name) & (df["호선"].astype(str) == line_name)].copy()
    if matched.empty:
        return matched

    return matched.sort_values(["날짜", "시간"]).tail(sample_size).reset_index(drop=True)


def calculate_mae(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_true - y_pred)))


def label_mae_status(avg_mae, best_avg_mae):
    if pd.isna(avg_mae):
        return "계산 실패"
    if best_avg_mae <= 0:
        return "최우수"

    ratio = avg_mae / best_avg_mae
    if ratio <= 1.10:
        return "최우수"
    if ratio <= 1.30:
        return "우수"
    if ratio <= 1.60:
        return "보통"
    return "개선 필요"


def render_comparison_tabs(
    tab4, tab5, selected_station, dt, selected_hour,
    temp, rain, snow, active_model, ALL_MODELS, le_station, LSTM_BASE_DF, STATIONS, LINE_OPTIONS
):
    base_avg = get_station_avg(selected_station, LSTM_BASE_DF)

    # ── 탭 4: 최적 탑승 시간 추천 ──────────────────────────────────────────────────
    with tab4:
        st.subheader("최적 탑승 시간 추천")
        st.caption("출발역과 도착역의 시간대별 혼잡도를 함께 비교해 가장 한산한 시간대를 찾습니다.")

        default_from = selected_station
        default_to = (
            find_station_by_name(STATIONS, "홍대입구", "2호선")
            or find_station_by_name(STATIONS, "홍대입구")
            or STATIONS[min(len(STATIONS) - 1, 1)]
        )

        col_from, col_to = st.columns(2)
        with col_from:
            from_station = select_line_station("출발", STATIONS, LINE_OPTIONS, default_from, "optimal_from")
        with col_to:
            to_station = select_line_station("도착", STATIONS, LINE_OPTIONS, default_to, "optimal_to")

        if st.button("최적 시간 찾기", type="primary"):
            with st.spinner("시간대별 혼잡도 분석 중..."):
                results = []
                for hr in range(5, 24):
                    b_from, _ = predict(from_station, dt, hr, temp, rain, snow, active_model, ALL_MODELS, le_station, LSTM_BASE_DF)
                    b_to, _ = predict(to_station, dt, hr, temp, rain, snow, active_model, ALL_MODELS, le_station, LSTM_BASE_DF)
                    cong_from, _, _ = get_congestion(b_from, from_station, avg=get_station_avg(from_station, LSTM_BASE_DF))
                    cong_to, _, _ = get_congestion(b_to, to_station, avg=get_station_avg(to_station, LSTM_BASE_DF))
                    results.append({
                        "시간": hr,
                        "출발역_승차": b_from,
                        "도착역_승차": b_to,
                        "출발_혼잡": cong_from,
                        "도착_혼잡": cong_to,
                        "합산": b_from + b_to,
                    })

                res_df = pd.DataFrame(results)
                best = res_df.loc[res_df["합산"].idxmin()]
                worst = res_df.loc[res_df["합산"].idxmax()]

            st.success(
                f"추천 탑승 시간: **{int(best['시간'])}시** "
                f"({format_station_label(from_station)} {int(best['출발역_승차']):,}명 / "
                f"{format_station_label(to_station)} {int(best['도착역_승차']):,}명, {active_model} 기준)"
            )
            st.error(
                f"최대 혼잡 시간: **{int(worst['시간'])}시** "
                f"({int(worst['출발역_승차']):,}명 / {int(worst['도착역_승차']):,}명)"
            )

            cur = res_df[res_df["시간"] == selected_hour].iloc[0]
            diff = int(cur["합산"]) - int(best["합산"])
            if diff > 0:
                st.info(f"현재 선택한 {selected_hour}시보다 {int(best['시간'])}시에 타면 합산 기준 약 {diff:,}명 더 한산합니다.")

            fig4 = go.Figure()
            fig4.add_trace(go.Bar(
                x=res_df["시간"],
                y=res_df["출발역_승차"],
                name=f"{format_station_label(from_station)} 출발",
                marker_color="#3b82f6",
                hovertemplate="%{x}시 %{y:,}명<extra></extra>",
            ))
            fig4.add_trace(go.Bar(
                x=res_df["시간"],
                y=res_df["도착역_승차"],
                name=f"{format_station_label(to_station)} 도착",
                marker_color="#10b981",
                hovertemplate="%{x}시 %{y:,}명<extra></extra>",
            ))
            fig4.add_vline(
                x=best["시간"],
                line_dash="dot",
                line_color="#6366f1",
                line_width=2,
                annotation_text="추천",
                annotation_font_color="#6366f1",
            )
            fig4.add_vline(
                x=selected_hour,
                line_dash="dot",
                line_color="#f59e0b",
                line_width=2,
                annotation_text="현재 선택",
                annotation_font_color="#f59e0b",
            )
            fig4.update_layout(
                title="시간대별 출발·도착역 혼잡도 비교",
                barmode="group",
                xaxis=dict(title="시간", tickmode="linear", tick0=5, dtick=1, ticksuffix="시"),
                yaxis=dict(title="승차인원 (명)", tickformat=","),
                plot_bgcolor="white",
                paper_bgcolor="white",
                height=380,
                margin=dict(l=0, r=0, t=50, b=0),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            )
            fig4.update_xaxes(showgrid=False)
            fig4.update_yaxes(showgrid=True, gridcolor="#f1f5f9")
            st.plotly_chart(fig4, use_container_width=True)

    # ── 탭 5: 모델별 예측치 비교 ──────────────────────────────────────────────────
    with tab5:
        st.subheader("모델별 예측치 비교")
        st.caption(f"{format_station_label(selected_station)} · {dt.date()} · {selected_hour}시 기준")

        model_list = [m_name for m_name, info in ALL_MODELS.items() if info.get("loaded", False)]
        comp_results = []

        for model_name in model_list:
            b_val, a_val = predict(
                selected_station,
                dt,
                selected_hour,
                temp,
                rain,
                snow,
                model_name=model_name,
                all_models=ALL_MODELS,
                le_station=le_station,
                lstm_base_df=LSTM_BASE_DF,
            )
            c_lbl, c_col, _ = get_congestion(b_val, selected_station, avg=base_avg)
            comp_results.append({
                "모델": model_name,
                "예상 승차 (명)": f"{b_val:,}",
                "예상 하차 (명)": f"{a_val:,}",
                "혼잡도 판정": f"<span style='color:{c_col}; font-weight:700;'>{c_lbl}</span>",
            })

        comp_df = pd.DataFrame(comp_results)
        st.write(comp_df.to_html(escape=False, index=False), unsafe_allow_html=True)

        st.divider()

        col_chart1, col_chart2 = st.columns(2)
        model_colors = {
            "LSTM": "#f43f5e",
            "LightGBM": "#10b981",
            "XGBoost": "#3b82f6",
            "RandomForest": "#8b5cf6",
        }

        with col_chart1:
            st.subheader("모델별 24시간 승차 예측")
            fig_board = go.Figure()
            for model_name in model_list:
                hrs, b_vals, _ = predict_day(
                    selected_station,
                    dt,
                    temp,
                    rain,
                    snow,
                    model_name=model_name,
                    all_models=ALL_MODELS,
                    le_station=le_station,
                    lstm_base_df=LSTM_BASE_DF,
                )
                fig_board.add_trace(go.Scatter(
                    x=hrs,
                    y=b_vals,
                    mode="lines+markers",
                    name=model_name,
                    line=dict(color=model_colors.get(model_name, "#64748b"), width=2.5),
                    marker=dict(size=5),
                    hovertemplate="%{x}시 %{y:,}명 승차<extra></extra>",
                ))
            fig_board.update_layout(
                xaxis=dict(title="시간", tickmode="linear", tick0=5, dtick=2, ticksuffix="시"),
                yaxis=dict(title="예상 승차 인원 (명)", tickformat=","),
                plot_bgcolor="white",
                paper_bgcolor="white",
                height=400,
                margin=dict(l=0, r=0, t=10, b=0),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            )
            fig_board.update_xaxes(showgrid=False)
            fig_board.update_yaxes(showgrid=True, gridcolor="#f1f5f9")
            st.plotly_chart(fig_board, use_container_width=True)

        with col_chart2:
            st.subheader("모델별 24시간 하차 예측")
            fig_alight = go.Figure()
            for model_name in model_list:
                hrs, _, a_vals = predict_day(
                    selected_station,
                    dt,
                    temp,
                    rain,
                    snow,
                    model_name=model_name,
                    all_models=ALL_MODELS,
                    le_station=le_station,
                    lstm_base_df=LSTM_BASE_DF,
                )
                fig_alight.add_trace(go.Scatter(
                    x=hrs,
                    y=a_vals,
                    mode="lines+markers",
                    name=model_name,
                    line=dict(color=model_colors.get(model_name, "#64748b"), width=2.5),
                    marker=dict(size=5),
                    hovertemplate="%{x}시 %{y:,}명 하차<extra></extra>",
                ))
            fig_alight.update_layout(
                xaxis=dict(title="시간", tickmode="linear", tick0=5, dtick=2, ticksuffix="시"),
                yaxis=dict(title="예상 하차 인원 (명)", tickformat=","),
                plot_bgcolor="white",
                paper_bgcolor="white",
                height=400,
                margin=dict(l=0, r=0, t=10, b=0),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            )
            fig_alight.update_xaxes(showgrid=False)
            fig_alight.update_yaxes(showgrid=True, gridcolor="#f1f5f9")
            st.plotly_chart(fig_alight, use_container_width=True)

        st.divider()
        st.subheader("정량 성능 비교")
        metric_sample_size = 48

        if st.button("MAE 상태 계산", key="calculate_model_metrics"):
            metric_sample = load_metric_sample(selected_station, metric_sample_size)
            if metric_sample.empty:
                st.warning("선택한 역의 정량 비교용 관측 데이터를 찾지 못했습니다.")
            else:
                with st.spinner("모델별 MAE를 계산하는 중..."):
                    metric_rows = []
                    for model_name in model_list:
                        try:
                            board_true = []
                            board_pred = []
                            alight_true = []
                            alight_pred = []

                            for _, row in metric_sample.iterrows():
                                sample_dt = pd.to_datetime(row["날짜"]).to_pydatetime()
                                b_val, a_val = predict(
                                    selected_station,
                                    sample_dt,
                                    int(row["시간"]),
                                    float(row["기온"]),
                                    float(row["강수량"]),
                                    float(row["적설"]),
                                    model_name=model_name,
                                    all_models=ALL_MODELS,
                                    le_station=le_station,
                                    lstm_base_df=LSTM_BASE_DF,
                                )
                                board_true.append(float(row["승차인원"]))
                                board_pred.append(float(b_val))
                                alight_true.append(float(row["하차인원"]))
                                alight_pred.append(float(a_val))

                            board_mae = calculate_mae(board_true, board_pred)
                            alight_mae = calculate_mae(alight_true, alight_pred)
                            metric_rows.append({
                                "모델": model_name,
                                "승차 MAE": board_mae,
                                "하차 MAE": alight_mae,
                                "_평균 MAE": (board_mae + alight_mae) / 2,
                                "상태": "",
                            })
                        except Exception:
                            metric_rows.append({
                                "모델": model_name,
                                "승차 MAE": np.nan,
                                "하차 MAE": np.nan,
                                "_평균 MAE": np.nan,
                                "상태": "계산 실패",
                            })

                valid_scores = [row["_평균 MAE"] for row in metric_rows if not pd.isna(row["_평균 MAE"])]
                best_avg_mae = min(valid_scores) if valid_scores else np.nan
                for row in metric_rows:
                    if row["상태"] != "계산 실패":
                        row["상태"] = label_mae_status(row["_평균 MAE"], best_avg_mae)

                metric_df = pd.DataFrame(metric_rows)[["모델", "승차 MAE", "하차 MAE", "상태"]]
                display_metric_df = metric_df.copy()
                for col in ["승차 MAE", "하차 MAE"]:
                    display_metric_df[col] = display_metric_df[col].map(lambda value: "-" if pd.isna(value) else f"{value:,.1f}")

                st.caption(f"{format_station_label(selected_station)} · 최근 {len(metric_sample):,}개 관측값 기준")
                st.dataframe(display_metric_df, use_container_width=True, hide_index=True)

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from core.predictor import predict, predict_day, get_congestion

@st.cache_data(show_spinner="📊 실제 가중치 기반 모델 오차율(MAE) 검증 데이터 분석 중...")
def calculate_dynamic_metrics(_all_models, _le_station, _lstm_base_df, seed=42):
    from sklearn.metrics import mean_absolute_error, r2_score
    try:
        df_all = pd.read_csv("data/processed/final_dataset_line2_230101-241231.csv")
        df_all['날짜'] = pd.to_datetime(df_all['날짜'])
        # Sample 100 rows from the end period of the dataset for validation
        test_sample = df_all[df_all['날짜'] >= '2024-12-15'].sample(100, random_state=seed).reset_index(drop=True)
    except Exception:
        return []

    results = []
    # Order: LSTM, XGBoost, LightGBM, RandomForest
    model_names = ["LSTM", "XGBoost", "LightGBM", "RandomForest"]
    
    for model_name in model_names:
        m_info = _all_models.get(model_name, {})
        if not m_info.get("loaded", False):
            continue
            
        y_true_board = []
        y_pred_board = []
        
        for _, row in test_sample.iterrows():
            station = row['역명']
            dt = row['날짜']
            hour = row['시간']
            temp = row['기온']
            rain = row['강수량']
            snow = row['적설']
            y_true = row['승차인원']
            
            try:
                b_pred, _ = predict(station, dt, hour, temp, rain, snow, model_name, _all_models, _le_station, _lstm_base_df)
                y_true_board.append(y_true)
                y_pred_board.append(b_pred)
            except Exception:
                pass
                
        if len(y_true_board) > 0:
            mae = mean_absolute_error(y_true_board, y_pred_board)
            r2 = r2_score(y_true_board, y_pred_board)
            
            r2_pct = max(0.0, r2 * 100)
            
            if r2 > 0.95 and mae < 150:
                reliability = "🥇 최우수"
            elif r2 > 0.90 and mae < 200:
                reliability = "🥈 우수"
            elif r2 > 0.70:
                reliability = "🥉 보통"
            else:
                reliability = "⚠️ 낮음 (과적합/오차 과다)"
                
            results.append({
                "예측 모델": model_name,
                "평균 오차 (MAE)": f"약 {mae:.1f} 명",
                "결정계수 (R² Score)": f"{r2_pct:.1f}% ({r2:.3f})",
                "모델 신뢰 수준": reliability,
                "mae_val": round(mae, 2)
            })
            
    return results


def render_comparison_tabs(tab4, tab5, selected_station, dt, selected_hour, temp, rain, snow, active_model, all_models, le_station, lstm_base_df, STATIONS):
    # ================================================================================
    # TAB 4: 최적 탑승 시간 추천
    # ================================================================================
    with tab4:
        st.subheader("⏰ 최적 탑승 시간 추천")

        col_from, col_to = st.columns(2)
        with col_from:
            from_station = st.selectbox("출발역", STATIONS, index=STATIONS.index("강남") if "강남" in STATIONS else 0, key="from")
        with col_to:
            to_station = st.selectbox("도착역", STATIONS, index=STATIONS.index("홍대입구") if "홍대입구" in STATIONS else 0, key="to")

        if st.button("최적 시간 찾기 🔍", type="primary"):
            with st.spinner("시간대별 혼잡도 분석 중..."):
                results = []
                for hr in range(5, 24):
                    b_f, _ = predict(from_station, dt, hr, temp, rain, snow, active_model, all_models, le_station, lstm_base_df)
                    b_t, _ = predict(to_station,   dt, hr, temp, rain, snow, active_model, all_models, le_station, lstm_base_df)
                    cong_f, _, _ = get_congestion(b_f, from_station)
                    cong_t, _, _ = get_congestion(b_t, to_station)
                    score = b_f + b_t
                    results.append({
                        "시간": hr, "출발역_승차": b_f, "도착역_승차": b_t,
                        "출발_혼잡": cong_f, "도착_혼잡": cong_t, "합산": score
                    })

                res_df = pd.DataFrame(results)
                best = res_df.loc[res_df["합산"].idxmin()]
                worst = res_df.loc[res_df["합산"].idxmax()]

            st.success(f"✅ **추천 탑승 시간: {int(best['시간'])}시** — 출발역 {int(best['출발역_승차']):,}명 / 도착역 {int(best['도착역_승차']):,}명 ({active_model} 기준)")
            st.error(f"⛔ **최대 혼잡 시간: {int(worst['시간'])}시** — 출발역 {int(worst['출발역_승차']):,}명 / 도착역 {int(worst['도착역_승차']):,}명 ({active_model} 기준)")

            cur = res_df[res_df["시간"] == selected_hour].iloc[0]
            diff = int(cur["합산"]) - int(best["합산"])
            if diff > 0:
                st.info(f"ℹ️ 현재 선택한 {selected_hour}시보다 **{int(best['시간'])}시**에 타면 합산 **{diff:,}명** 더 한산해요.")

            fig4 = go.Figure()
            fig4.add_trace(go.Bar(
                x=res_df["시간"], y=res_df["출발역_승차"],
                name=f"{from_station} 출발", marker_color="#3b82f6",
                hovertemplate="%{x}시 %{y:,}명<extra></extra>",
            ))
            fig4.add_trace(go.Bar(
                x=res_df["시간"], y=res_df["도착역_승차"],
                name=f"{to_station} 도착", marker_color="#10b981",
                hovertemplate="%{x}시 %{y:,}명<extra></extra>",
            ))
            fig4.add_vline(x=best["시간"], line_dash="dot", line_color="#6366f1", line_width=2,
                           annotation_text="추천", annotation_font_color="#6366f1")
            fig4.add_vline(x=selected_hour, line_dash="dot", line_color="#f59e0b", line_width=2,
                           annotation_text="현재 선택", annotation_font_color="#f59e0b")

            fig4.update_layout(
                title="시간대별 출발·도착역 혼잡도 비교", barmode="group",
                xaxis=dict(title="시간", tickmode="linear", tick0=5, dtick=1, ticksuffix="시"),
                yaxis=dict(title="승차인원 (명)", tickformat=","),
                plot_bgcolor="white", paper_bgcolor="white",
                height=380, margin=dict(l=0, r=0, t=50, b=0),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            )
            fig4.update_xaxes(showgrid=False)
            fig4.update_yaxes(showgrid=True, gridcolor="#f1f5f9")
            st.plotly_chart(fig4, use_container_width=True)

        st.divider()
        st.caption("💡 최적 탑승 시간은 출발역 + 도착역 혼잡도 합산 기준으로 계산됩니다.")

    # ================================================================================
    # TAB 5: AI 모델별 예측치 비교
    # ================================================================================
    with tab5:
        st.subheader("🤖 개발 모델별 실시간 예측 결과 비교 분석")
        st.markdown("팀원들이 개발한 4개의 예측 모델(XGBoost, LightGBM, RandomForest, LSTM)의 예측 수치와 혼잡도 수준을 대조하고 차이를 시각화합니다.")
        
        col_comp_info, col_comp_metric = st.columns([3, 2])
        with col_comp_info:
            st.info(f"🚉 비교 대상역: **{selected_station}역** · **{selected_hour}시** 기준")
        
        model_list = [m_name for m_name, info in all_models.items() if info.get("loaded", False)]
        comp_results = []
        
        for m_name in model_list:
            b_val, a_val = predict(selected_station, dt, selected_hour, temp, rain, snow, model_name=m_name, all_models=all_models, le_station=le_station, lstm_base_df=lstm_base_df)
            c_lbl, c_col, _ = get_congestion(b_val, selected_station)
            
            comp_results.append({
                "모델": m_name,
                "예상 승차 (명)": b_val,
                "예상 하차 (명)": a_val,
                "혼잡도 판정": f"<span style='color:{c_col}; font-weight:700;'>{c_lbl}</span>"
            })
            
        comp_df = pd.DataFrame(comp_results)
        
        with col_comp_info:
            st.write(comp_df.to_html(escape=False, index=False), unsafe_allow_html=True)
            
        st.markdown("---")
        col_chart1, col_chart2 = st.columns(2)
        
        model_colors = {
            "XGBoost": "#3b82f6", "LightGBM": "#10b981", "RandomForest": "#8b5cf6", "LSTM": "#f43f5e"
        }
        
        with col_chart1:
            st.subheader("📈 모델별 24시간 승차 예측 패턴")
            fig_comp_board = go.Figure()
            for m_name in model_list:
                hrs, b_vals, _ = predict_day(selected_station, dt, temp, rain, snow, model_name=m_name, all_models=all_models, le_station=le_station, lstm_base_df=lstm_base_df)
                fig_comp_board.add_trace(go.Scatter(
                    x=hrs, y=b_vals, mode="lines+markers", name=m_name,
                    line=dict(color=model_colors[m_name], width=2.5), marker=dict(size=5),
                    hovertemplate="%{x}시 %{y:,}명 승차<extra></extra>"
                ))
                
            fig_comp_board.update_layout(
                xaxis=dict(title="시간 (시)", tickmode="linear", tick0=5, dtick=2, ticksuffix="시"),
                yaxis=dict(title="예상 승차 인원 (명)", tickformat=","),
                plot_bgcolor="white", paper_bgcolor="white",
                height=400, margin=dict(l=0, r=0, t=10, b=0),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            )
            fig_comp_board.update_xaxes(showgrid=False)
            fig_comp_board.update_yaxes(showgrid=True, gridcolor="#f1f5f9")
            st.plotly_chart(fig_comp_board, use_container_width=True)
            
        with col_chart2:
            st.subheader("📉 모델별 24시간 하차 예측 패턴")
            fig_comp_alight = go.Figure()
            for m_name in model_list:
                hrs, _, a_vals = predict_day(selected_station, dt, temp, rain, snow, model_name=m_name, all_models=all_models, le_station=le_station, lstm_base_df=lstm_base_df)
                fig_comp_alight.add_trace(go.Scatter(
                    x=hrs, y=a_vals, mode="lines+markers", name=m_name,
                    line=dict(color=model_colors[m_name], width=2.5), marker=dict(size=5),
                    hovertemplate="%{x}시 %{y:,}명 하차<extra></extra>"
                ))
                
            fig_comp_alight.update_layout(
                xaxis=dict(title="시간 (시)", tickmode="linear", tick0=5, dtick=2, ticksuffix="시"),
                yaxis=dict(title="예상 하차 인원 (명)", tickformat=","),
                plot_bgcolor="white", paper_bgcolor="white",
                height=400, margin=dict(l=0, r=0, t=10, b=0),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            )
            fig_comp_alight.update_xaxes(showgrid=False)
            fig_comp_alight.update_yaxes(showgrid=True, gridcolor="#f1f5f9")
            st.plotly_chart(fig_comp_alight, use_container_width=True)
            
        st.divider()
        st.caption("💡 현재 정상적으로 로드되어 가동 중인 2호선 승하차 예측 AI 모델들의 실시간 예측 수치를 비교 대조합니다.")
        
        # ── 신설: AI 모델별 성능 분석 및 오답률(오차 평가) 리포트 섹션 ────────────────────────
        st.markdown("---")
        st.subheader("📊 AI 모델별 교차 검증 오차율 (오답률) 분석")
        if "test_seed" not in st.session_state:
            st.session_state.test_seed = 42

        col_metric_desc, col_metric_btn = st.columns([3, 1])
        with col_metric_desc:
            st.markdown(f"팀 프로젝트 연구 및 개발 과정에서 검증용 테스트 데이터셋으로 도출한 각 머신러닝/딥러닝 모델별 평균 절대 오차(MAE) 및 결정계수($R^2$ Score) 검증 성적표입니다. (현재 검증 데이터 시드: **{st.session_state.test_seed}**)")
        with col_metric_btn:
            if st.button("🔄 무작위 재시험", help="새로운 100개 무작위 샘플 데이터를 기반으로 성능 오차율을 실시간으로 재연산합니다."):
                import random
                st.session_state.test_seed = random.randint(1, 10000)
                st.rerun()
        
        dynamic_results = calculate_dynamic_metrics(all_models, le_station, lstm_base_df, seed=st.session_state.test_seed)
        
        if dynamic_results:
            eval_df = pd.DataFrame(dynamic_results)
            # Remove mae_val from displayed table columns
            display_df = eval_df.drop(columns=["mae_val"])
            
            col_metric_tbl, col_metric_chart = st.columns([1, 1])
            
            with col_metric_tbl:
                st.markdown(f"##### 🏆 검증 데이터셋(Test Set) 모델 평가 결과 (시드 {st.session_state.test_seed} 연산)")
                st.write(display_df.to_html(escape=False, index=False), unsafe_allow_html=True)
                st.caption(f"💡 MAE(평균 절대 오차)는 실제 승하차 승객 수와 모델 예측치 간의 오차 인원 평균으로, 수치가 낮을수록 정확합니다. (최근 2주 데이터 중 100행 샘플)")
                
            with col_metric_chart:
                st.markdown("##### 📉 모델별 평균 절대 오차(MAE) 비교 시각화")
                fig_mae = go.Figure()
                
                # 모델별 색상 매핑
                m_colors = {"LSTM": "#f43f5e", "XGBoost": "#3b82f6", "LightGBM": "#10b981", "RandomForest": "#8b5cf6"}
                bar_colors = [m_colors.get(m, "#64748b") for m in eval_df["예측 모델"]]
                
                fig_mae.add_trace(go.Bar(
                    x=eval_df["예측 모델"],
                    y=eval_df["mae_val"],
                    marker_color=bar_colors,
                    hovertemplate="%{x} MAE: %{y}명<extra></extra>",
                    text=[f"{v}명" for v in eval_df["mae_val"]],
                    textposition="auto"
                ))
                
                fig_mae.update_layout(
                    xaxis=dict(title="평가 대상 AI 모델"),
                    yaxis=dict(title="평균 오차 인원 (명)"),
                    plot_bgcolor="white",
                    paper_bgcolor="white",
                    height=250,
                    margin=dict(l=10, r=10, t=10, b=10),
                    showlegend=False
                )
                fig_mae.update_yaxes(showgrid=True, gridcolor="#f1f5f9")
                st.plotly_chart(fig_mae, use_container_width=True)
        else:
            st.error("모델 성능 검증 데이터를 불러올 수 없거나 연산 중 오류가 발생했습니다.")



import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from core.predictor import predict, predict_day, get_congestion

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

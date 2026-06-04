import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from core.config import STATION_AVG
from core.predictor import predict, predict_day, get_congestion

def render_forecast_tabs(tab1, tab2, tab3, selected_station, dt, selected_hour, temp, rain, snow, active_model, all_models, le_station, lstm_base_df, STATIONS):
    # ================================================================================
    # TAB 1: 시간대별 예측
    # ================================================================================
    with tab1:
        col_info, col_metric = st.columns([3, 2])
        with col_info:
            st.subheader(f"🚉 {selected_station}역 · {dt.strftime('%Y년 %m월 %d일')}")
            day_names = ["월", "화", "수", "목", "금", "토", "일"]
            day_str = day_names[dt.weekday()]
            st.caption(f"{day_str}요일 · 기온 {temp}°C · 강수량 {rain}mm · 적설 {snow}mm")

        board_now, alight_now = predict(selected_station, dt, selected_hour, temp, rain, snow, active_model, all_models, le_station, lstm_base_df)
        cong_label, cong_color, cong_bg = get_congestion(board_now, selected_station)

        with col_metric:
            st.markdown(f"""
            <div style="background:{cong_bg}; border-radius:10px; padding:12px 16px; text-align:center; border:1px solid {cong_color}33;">
                <div style="font-size:12px; color:{cong_color}; margin-bottom:4px;">{selected_hour}시 혼잡도 ({active_model})</div>
                <div style="font-size:22px; font-weight:700; color:{cong_color};">{cong_label}</div>
            </div>
            """, unsafe_allow_html=True)

        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric("🔼 예상 승차인원", f"{board_now:,}명", help="선택한 시간대 예상 승차인원")
        with m2:
            st.metric("🔽 예상 하차인원", f"{alight_now:,}명", help="선택한 시간대 예상 하차인원")
        with m3:
            avg = STATION_AVG.get(selected_station, 1500)
            diff_pct = round((board_now - avg) / avg * 100) if avg > 0 else 0
            st.metric("📊 평균 대비", f"{diff_pct:+}%", delta=f"{'↑ 평균 초과' if diff_pct > 0 else '↓ 평균 이하'}")

        st.divider()

        hours, boards, alights = predict_day(selected_station, dt, temp, rain, snow, active_model, all_models, le_station, lstm_base_df)
        fig = go.Figure()

        fig.add_vrect(
            x0=6.5, x1=9.5, fillcolor="rgba(59, 130, 246, 0.1)", line_width=0,
            annotation_text="출근 피크", annotation_position="top left",
            annotation_font_size=11, annotation_font_color="#3b82f6"
        )
        fig.add_vrect(
            x0=17.5, x1=20.5, fillcolor="rgba(245, 158, 11, 0.1)", line_width=0,
            annotation_text="퇴근 피크", annotation_position="top left",
            annotation_font_size=11, annotation_font_color="#f59e0b"
        )
        fig.add_vline(
            x=selected_hour, line_dash="dot", line_color="#6366f1", line_width=2,
            annotation_text=f"선택: {selected_hour}시", annotation_position="top right",
            annotation_font_color="#6366f1"
        )

        fig.add_trace(go.Scatter(
            x=hours, y=boards, mode="lines+markers", name="승차인원",
            line=dict(color="#3b82f6", width=2.5),
            marker=dict(
                size=[10 if h == selected_hour else 6 for h in hours],
                color=["#6366f1" if h == selected_hour else "#3b82f6" for h in hours],
            ),
            hovertemplate="%{x}시 승차 %{y:,}명<extra></extra>"
        ))
        fig.add_trace(go.Scatter(
            x=hours, y=alights, mode="lines+markers", name="하차인원",
            line=dict(color="#10b981", width=2.5),
            marker=dict(
                size=[10 if h == selected_hour else 6 for h in hours],
                color=["#059669" if h == selected_hour else "#10b981" for h in hours],
            ),
            hovertemplate="%{x}시 하차 %{y:,}명<extra></extra>"
        ))

        fig.update_layout(
            title=f"{selected_station}역 시간대별 승하차 예측 ({dt.date()})",
            xaxis=dict(title="시간", tickmode="linear", tick0=5, dtick=1, ticksuffix="시"),
            yaxis=dict(title="인원 수 (명)", tickformat=","),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            hovermode="x unified", plot_bgcolor="white", paper_bgcolor="white",
            margin=dict(l=0, r=0, t=50, b=0), height=420,
        )
        fig.update_xaxes(showgrid=True, gridcolor="#f1f5f9")
        fig.update_yaxes(showgrid=True, gridcolor="#f1f5f9")
        st.plotly_chart(fig, use_container_width=True)

        max_board_hr = hours[np.argmax(boards)]
        max_board = max(boards)
        max_alight_hr = hours[np.argmax(alights)]
        max_alight = max(alights)

        c1, c2 = st.columns(2)
        c1.info(f"🔵 **승차 피크 시간대**: {max_board_hr}시 ({max_board:,}명)")
        c2.success(f"🟢 **하차 피크 시간대**: {max_alight_hr}시 ({max_alight:,}명)")

    # ================================================================================
    # TAB 2: 날씨 영향도 비교
    # ================================================================================
    with tab2:
        st.subheader(f"🌦 {selected_station}역 날씨 조건별 비교")
        st.caption(f"{dt.date()} · {selected_hour}시 · 기온 {temp}°C 기준")

        b_clear, a_clear = predict(selected_station, dt, selected_hour, temp, 0.0, 0.0, active_model, all_models, le_station, lstm_base_df)
        b_rain, a_rain = predict(selected_station, dt, selected_hour, temp, 10.0, 0.0, active_model, all_models, le_station, lstm_base_df)
        b_snow, a_snow = predict(selected_station, dt, selected_hour, temp - 5, 0.0, 5.0, active_model, all_models, le_station, lstm_base_df)

        c1, c2, c3 = st.columns(3)
        conditions = [
            ("☀️ 맑음", b_clear, a_clear, "#fef9c3", "#854d0e", "0mm 강수"),
            ("🌧 비", b_rain, a_rain, "#dbeafe", "#1e40af", "10mm 강수"),
            ("❄️ 눈", b_snow, a_snow, "#e0f2fe", "#0c4a6e", "5mm 적설"),
        ]

        for col, (label, b, a, bg, color, sub) in zip([c1, c2, c3], conditions):
            cong, ccol, cbg = get_congestion(b, selected_station)
            col.markdown(f"""
            <div style="background:{bg}; border-radius:12px; padding:16px; text-align:center; border:1px solid {color}22;">
                <div style="font-size:20px; margin-bottom:8px;">{label}</div>
                <div style="font-size:11px; color:{color}; margin-bottom:12px;">{sub}</div>
                <div style="font-size:24px; font-weight:700; color:{color}; margin-bottom:4px;">{b:,}명</div>
                <div style="font-size:12px; color:{color}; margin-bottom:8px;">승차인원</div>
                <div style="font-size:18px; font-weight:500; color:{color};">{a:,}명</div>
                <div style="font-size:12px; color:{color}; margin-bottom:8px;">하차인원</div>
                <span style="background:{cbg}; color:{ccol}; padding:3px 10px; border-radius:999px; font-size:12px;">{cong}</span>
            </div>
            """, unsafe_allow_html=True)

        st.divider()

        st.subheader("하루 전체 날씨 조건별 승차 비교")
        hours_s, b_c, _ = predict_day(selected_station, dt, temp, 0.0, 0.0, active_model, all_models, le_station, lstm_base_df)
        _, b_r, _ = predict_day(selected_station, dt, temp, 10.0, 0.0, active_model, all_models, le_station, lstm_base_df)
        _, b_n, _ = predict_day(selected_station, dt, temp - 5, 0.0, 5.0, active_model, all_models, le_station, lstm_base_df)

        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=hours_s, y=b_c, name="맑음 시나리오", line=dict(color="#f59e0b", width=2), mode="lines"))
        fig2.add_trace(go.Scatter(x=hours_s, y=b_r, name="비 시나리오", line=dict(color="#3b82f6", width=2), mode="lines"))
        fig2.add_trace(go.Scatter(x=hours_s, y=b_n, name="눈 시나리오", line=dict(color="#06b6d4", width=2, dash="dot"), mode="lines"))

        max_diff_rain = max([c - r for c, r in zip(b_c, b_r)])
        max_diff_snow = max([c - n for c, n in zip(b_c, b_n)])

        fig2.update_layout(
            title=f"맑음 vs 비 vs 눈 — 최대 차이: 비 -{max_diff_rain:,}명 / 눈 -{max_diff_snow:,}명",
            xaxis=dict(title="시간", tickmode="linear", tick0=5, dtick=1, ticksuffix="시"),
            yaxis=dict(title="승차인원 (명)", tickformat=","),
            plot_bgcolor="white", paper_bgcolor="white",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            height=380, margin=dict(l=0, r=0, t=50, b=0),
        )
        fig2.update_xaxes(showgrid=True, gridcolor="#f1f5f9")
        fig2.update_yaxes(showgrid=True, gridcolor="#f1f5f9")
        st.plotly_chart(fig2, use_container_width=True)

        col_a, col_b = st.columns(2)
        rain_diff = round(np.mean([c - r for c, r in zip(b_c, b_r)]))
        snow_diff = round(np.mean([c - n for c, n in zip(b_c, b_n)]))
        col_a.warning(f"🌧 강수 시나리오: 맑은 날 대비 평균 {rain_diff:,}명 승객 감소")
        col_b.info(f"❄️ 강설 시나리오: 맑은 날 대비 평균 {snow_diff:,}명 승객 감소")

    # ================================================================================
    # TAB 3: 역별 혼잡도 랭킹
    # ================================================================================
    with tab3:
        st.subheader("🏆 2호선 전체 역 혼잡도 랭킹")
        st.caption(f"{dt.date()} · {selected_hour}시 · 기온 {temp}°C · 강수량 {rain}mm")

        with st.spinner("2호선 전체 역 예측 중..."):
            rankings = []
            for stn in STATIONS:
                b, a = predict(stn, dt, selected_hour, temp, rain, snow, active_model, all_models, le_station, lstm_base_df)
                cong, _, _ = get_congestion(b, stn)
                rankings.append({"역명": stn, "승차": b, "하차": a, "혼잡도": cong})

            rank_df = pd.DataFrame(rankings).sort_values("승차", ascending=False).reset_index(drop=True)

        col_top, col_bot = st.columns(2)
        with col_top:
            st.markdown("**🔴 혼잡 TOP 5**")
            for i, row in rank_df.head(5).iterrows():
                cong, ccol, cbg = get_congestion(row["승차"], row["역명"])
                st.markdown(f"""
                <div style="background:{cbg}; border-radius:8px; padding:10px 14px; margin-bottom:6px;
                            border:1px solid {ccol}33; display:flex; justify-content:space-between; align-items:center;">
                    <span style="font-weight:500; color:{ccol};">#{i+1} {row['역명']}</span>
                    <span style="color:{ccol}; font-size:13px;">{row['승차']:,}명 &nbsp;
                        <span style="background:{cbg}; border:1px solid {ccol}55; padding:2px 8px; border-radius:999px; font-size:11px;">{cong}</span>
                    </span>
                </div>
                """, unsafe_allow_html=True)

        with col_bot:
            st.markdown("**🟢 한산 TOP 5**")
            for i, row in rank_df.tail(5).iloc[::-1].iterrows():
                cong, ccol, cbg = get_congestion(row["승차"], row["역명"])
                st.markdown(f"""
                <div style="background:{cbg}; border-radius:8px; padding:10px 14px; margin-bottom:6px;
                            border:1px solid {ccol}33; display:flex; justify-content:space-between; align-items:center;">
                    <span style="font-weight:500; color:{ccol};">{row['역명']}</span>
                    <span style="color:{ccol}; font-size:13px;">{row['승차']:,}명 &nbsp;
                        <span style="background:{cbg}; border:1px solid {ccol}55; padding:2px 8px; border-radius:999px; font-size:11px;">{cong}</span>
                    </span>
                </div>
                """, unsafe_allow_html=True)

        st.divider()

        fig3 = go.Figure()
        colors = []
        for _, row in rank_df.iterrows():
            _, ccol, _ = get_congestion(row["승차"], row["역명"])
            colors.append(ccol)

        fig3.add_trace(go.Bar(
            x=rank_df["역명"], y=rank_df["승차"], marker_color=colors,
            hovertemplate="%{x} 승차 %{y:,}명<extra></extra>", name="승차인원"
        ))
        fig3.update_layout(
            title=f"2호선 전체 역 승차인원 ({selected_hour}시)",
            xaxis=dict(title="역명", tickangle=-45, tickfont=dict(size=10)),
            yaxis=dict(title="승차인원 (명)", tickformat=","),
            plot_bgcolor="white", paper_bgcolor="white",
            height=380, margin=dict(l=0, r=0, t=50, b=80), showlegend=False,
        )
        fig3.update_xaxes(showgrid=False)
        fig3.update_yaxes(showgrid=True, gridcolor="#f1f5f9")
        st.plotly_chart(fig3, use_container_width=True)

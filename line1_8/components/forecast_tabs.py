import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core.config import format_station_label
from core.predictor import predict, predict_day, get_congestion, get_station_avg


def render_forecast_tabs(
    tab1, tab2, tab3, selected_station, dt, selected_hour,
    temp, rain, snow, active_model, ALL_MODELS, le_station, LSTM_BASE_DF, STATIONS
):
    base_avg = get_station_avg(selected_station, LSTM_BASE_DF)

    # ── 탭 1: 시간대 예측 ────────────────────────────────────────────────────────
    with tab1:
        col_info, col_metric = st.columns([3, 2])
        with col_info:
            st.subheader(f"{format_station_label(selected_station)}")
            st.caption(f"{dt.strftime('%Y-%m-%d')} · {selected_hour}시 · 기온 {temp}℃ · 강수량 {rain}mm · 적설 {snow}mm")

        board_now, alight_now = predict(
            selected_station,
            dt,
            selected_hour,
            temp,
            rain,
            snow,
            active_model,
            ALL_MODELS,
            le_station,
            LSTM_BASE_DF,
        )
        cong_label, cong_color, cong_bg = get_congestion(board_now, selected_station, avg=base_avg)

        with col_metric:
            st.markdown(
                f"""
                <div style="background:{cong_bg}; border-radius:10px; padding:12px 16px; text-align:center; border:1px solid {cong_color}33;">
                    <div style="font-size:12px; color:{cong_color}; margin-bottom:4px;">{selected_hour}시 혼잡도</div>
                    <div style="font-size:22px; font-weight:700; color:{cong_color};">{cong_label}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric("예측 승차인원", f"{board_now:,}명")
        with m2:
            st.metric("예측 하차인원", f"{alight_now:,}명")
        with m3:
            diff_pct = round((board_now - base_avg) / base_avg * 100) if base_avg > 0 else 0
            st.metric("역 평균 대비", f"{diff_pct:+}%", delta=f"{'평균 초과' if diff_pct > 0 else '평균 이하'}")

        st.divider()

        hours, boards, alights = predict_day(
            selected_station,
            dt,
            temp,
            rain,
            snow,
            active_model,
            ALL_MODELS,
            le_station,
            LSTM_BASE_DF,
        )

        fig = go.Figure()
        fig.add_vrect(
            x0=6.5,
            x1=9.5,
            fillcolor="rgba(59, 130, 246, 0.1)",
            line_width=0,
            annotation_text="출근 피크",
            annotation_position="top left",
            annotation_font_size=11,
            annotation_font_color="#3b82f6",
        )
        fig.add_vrect(
            x0=17.5,
            x1=20.5,
            fillcolor="rgba(245, 158, 11, 0.1)",
            line_width=0,
            annotation_text="퇴근 피크",
            annotation_position="top left",
            annotation_font_size=11,
            annotation_font_color="#f59e0b",
        )
        fig.add_vline(
            x=selected_hour,
            line_dash="dot",
            line_color="#6366f1",
            line_width=2,
            annotation_text=f"선택: {selected_hour}시",
            annotation_position="top right",
            annotation_font_color="#6366f1",
        )
        fig.add_trace(go.Scatter(
            x=hours,
            y=boards,
            mode="lines+markers",
            name="승차인원",
            line=dict(color="#3b82f6", width=2.5),
            marker=dict(size=[10 if h == selected_hour else 6 for h in hours], color=["#6366f1" if h == selected_hour else "#3b82f6" for h in hours]),
            hovertemplate="%{x}시 승차 %{y:,}명<extra></extra>",
        ))
        fig.add_trace(go.Scatter(
            x=hours,
            y=alights,
            mode="lines+markers",
            name="하차인원",
            line=dict(color="#10b981", width=2.5),
            marker=dict(size=[10 if h == selected_hour else 6 for h in hours], color=["#059669" if h == selected_hour else "#10b981" for h in hours]),
            hovertemplate="%{x}시 하차 %{y:,}명<extra></extra>",
        ))
        fig.update_layout(
            title=f"{format_station_label(selected_station)} 시간대별 예측",
            xaxis=dict(title="시간", tickmode="linear", tick0=5, dtick=1, ticksuffix="시"),
            yaxis=dict(title="인원 (명)", tickformat=","),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            hovermode="x unified",
            plot_bgcolor="white",
            paper_bgcolor="white",
            margin=dict(l=0, r=0, t=50, b=0),
            height=420,
        )
        fig.update_xaxes(showgrid=True, gridcolor="#f1f5f9")
        fig.update_yaxes(showgrid=True, gridcolor="#f1f5f9")
        st.plotly_chart(fig, use_container_width=True)

        max_board_hr = hours[int(np.argmax(boards))]
        max_board = max(boards)
        max_alight_hr = hours[int(np.argmax(alights))]
        max_alight = max(alights)

        c1, c2 = st.columns(2)
        c1.info(f"승차 피크: {max_board_hr}시 ({max_board:,}명)")
        c2.success(f"하차 피크: {max_alight_hr}시 ({max_alight:,}명)")

    # ── 탭 2: 날씨 비교 ────────────────────────────────────────────────────────
    with tab2:
        st.subheader("날씨 조건별 비교")
        st.caption(f"{format_station_label(selected_station)} · {dt.date()} · {selected_hour}시 기준")

        b_clear, a_clear = predict(selected_station, dt, selected_hour, temp, 0.0, 0.0, active_model, ALL_MODELS, le_station, LSTM_BASE_DF)
        b_rain, a_rain = predict(selected_station, dt, selected_hour, temp, 10.0, 0.0, active_model, ALL_MODELS, le_station, LSTM_BASE_DF)
        b_snow, a_snow = predict(selected_station, dt, selected_hour, temp - 5, 0.0, 5.0, active_model, ALL_MODELS, le_station, LSTM_BASE_DF)

        c1, c2, c3 = st.columns(3)
        scenarios = [
            ("맑음", b_clear, a_clear, "#fef9c3", "#854d0e", "강수 0mm"),
            ("비", b_rain, a_rain, "#dbeafe", "#1e40af", "강수 10mm"),
            ("눈", b_snow, a_snow, "#e0f2fe", "#0c4a6e", "적설 5mm"),
        ]

        for col, (label, b, a, bg, color, sub) in zip([c1, c2, c3], scenarios):
            cong, ccol, cbg = get_congestion(b, selected_station, avg=base_avg)
            col.markdown(
                f"""
                <div style="background:{bg}; border-radius:10px; padding:16px; text-align:center; border:1px solid {color}22;">
                    <div style="font-size:20px; margin-bottom:8px;">{label}</div>
                    <div style="font-size:11px; color:{color}; margin-bottom:12px;">{sub}</div>
                    <div style="font-size:24px; font-weight:700; color:{color}; margin-bottom:4px;">{b:,}명</div>
                    <div style="font-size:12px; color:{color}; margin-bottom:8px;">승차</div>
                    <div style="font-size:18px; font-weight:500; color:{color};">{a:,}명</div>
                    <div style="font-size:12px; color:{color}; margin-bottom:8px;">하차</div>
                    <span style="background:{cbg}; color:{ccol}; padding:3px 10px; border-radius:999px; font-size:12px;">{cong}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.divider()

        hours_s, b_c, _ = predict_day(selected_station, dt, temp, 0.0, 0.0, active_model, ALL_MODELS, le_station, LSTM_BASE_DF)
        _, b_r, _ = predict_day(selected_station, dt, temp, 10.0, 0.0, active_model, ALL_MODELS, le_station, LSTM_BASE_DF)
        _, b_n, _ = predict_day(selected_station, dt, temp - 5, 0.0, 5.0, active_model, ALL_MODELS, le_station, LSTM_BASE_DF)

        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=hours_s, y=b_c, name="맑음", line=dict(color="#f59e0b", width=2), mode="lines"))
        fig2.add_trace(go.Scatter(x=hours_s, y=b_r, name="비", line=dict(color="#3b82f6", width=2), mode="lines"))
        fig2.add_trace(go.Scatter(x=hours_s, y=b_n, name="눈", line=dict(color="#06b6d4", width=2, dash="dot"), mode="lines"))
        fig2.update_layout(
            title="날씨 조건별 승차인원 변화",
            xaxis=dict(title="시간", tickmode="linear", tick0=5, dtick=1, ticksuffix="시"),
            yaxis=dict(title="승차인원 (명)", tickformat=","),
            plot_bgcolor="white",
            paper_bgcolor="white",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            height=380,
            margin=dict(l=0, r=0, t=50, b=0),
        )
        fig2.update_xaxes(showgrid=True, gridcolor="#f1f5f9")
        fig2.update_yaxes(showgrid=True, gridcolor="#f1f5f9")
        st.plotly_chart(fig2, use_container_width=True)

    # ── 탭 3: 역별 랭킹 ────────────────────────────────────────────────────────
    with tab3:
        st.subheader("전체 역 랭킹")
        st.caption(f"{format_station_label(selected_station)}와 같은 날짜·시간 조건에서 1-8호선 전체 역을 비교합니다.")

        with st.spinner("전체 역 예측값을 계산하는 중..."):
            rankings = []
            for stn in STATIONS:
                b, a = predict(stn, dt, selected_hour, temp, rain, snow, active_model, ALL_MODELS, le_station, LSTM_BASE_DF)
                avg = get_station_avg(stn, LSTM_BASE_DF)
                cong, _, _ = get_congestion(b, stn, avg=avg)
                rankings.append({"역명": stn, "승차": b, "하차": a, "혼잡도": cong})

            rank_df = pd.DataFrame(rankings).sort_values("승차", ascending=False).reset_index(drop=True)

        col_top, col_bot = st.columns(2)
        with col_top:
            st.markdown("**상위 10개 역**")
            for i, row in rank_df.head(10).iterrows():
                avg = get_station_avg(row["역명"], LSTM_BASE_DF)
                cong, ccol, cbg = get_congestion(row["승차"], row["역명"], avg=avg)
                st.markdown(
                    f"""
                    <div style="background:{cbg}; border-radius:8px; padding:10px 14px; margin-bottom:6px; border:1px solid {ccol}33; display:flex; justify-content:space-between; align-items:center;">
                        <span style="font-weight:500; color:{ccol};">#{i+1} {format_station_label(row['역명'])}</span>
                        <span style="color:{ccol}; font-size:13px;">{row['승차']:,}명 <span style="background:{cbg}; border:1px solid {ccol}55; padding:2px 8px; border-radius:999px; font-size:11px;">{cong}</span></span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        with col_bot:
            st.markdown("**하위 10개 역**")
            for i, row in rank_df.tail(10).iloc[::-1].iterrows():
                avg = get_station_avg(row["역명"], LSTM_BASE_DF)
                cong, ccol, cbg = get_congestion(row["승차"], row["역명"], avg=avg)
                st.markdown(
                    f"""
                    <div style="background:{cbg}; border-radius:8px; padding:10px 14px; margin-bottom:6px; border:1px solid {ccol}33; display:flex; justify-content:space-between; align-items:center;">
                        <span style="font-weight:500; color:{ccol};">{format_station_label(row['역명'])}</span>
                        <span style="color:{ccol}; font-size:13px;">{row['승차']:,}명 <span style="background:{cbg}; border:1px solid {ccol}55; padding:2px 8px; border-radius:999px; font-size:11px;">{cong}</span></span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        st.divider()

        fig3 = go.Figure()
        colors = []
        for _, row in rank_df.head(25).iterrows():
            avg = get_station_avg(row["역명"], LSTM_BASE_DF)
            _, ccol, _ = get_congestion(row["승차"], row["역명"], avg=avg)
            colors.append(ccol)

        fig3.add_trace(go.Bar(
            x=rank_df.head(25)["역명"].map(format_station_label),
            y=rank_df.head(25)["승차"],
            marker_color=colors,
            hovertemplate="%{x} 승차 %{y:,}명<extra></extra>",
            name="승차인원",
        ))
        fig3.update_layout(
            title=f"상위 25개 역 승차인원 ({selected_hour}시)",
            xaxis=dict(title="역명", tickangle=-45, tickfont=dict(size=10)),
            yaxis=dict(title="승차인원 (명)", tickformat=","),
            plot_bgcolor="white",
            paper_bgcolor="white",
            height=380,
            margin=dict(l=0, r=0, t=50, b=80),
            showlegend=False,
        )
        fig3.update_xaxes(showgrid=False)
        fig3.update_yaxes(showgrid=True, gridcolor="#f1f5f9")
        st.plotly_chart(fig3, use_container_width=True)

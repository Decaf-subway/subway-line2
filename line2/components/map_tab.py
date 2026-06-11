import streamlit as st
import numpy as np
import plotly.graph_objects as go
from core.config import MAIN_LINE, SUNGSU_BRANCH, SINDORIM_BRANCH, TRANSFER_INFO
from services.subway_api import get_realtime_train_positions, get_realtime_station_arrival

def render_map_tab(selected_station, seoul_key):
    st.subheader("🚇 2호선 실시간 열차 위치 관제 모니터")
    
    col_ref, col_lbl = st.columns([4, 1])
    with col_ref:
        st.caption("서울 열린데이터광장 실시간 열차 위치 API 연동")
    with col_lbl:
        if st.button("실시간 위치 새로고침 🔄"):
            st.rerun()

    # 2호선 51개 역에 대해 2차원 원형 좌표계(x, y) 구축
    N_main = len(MAIN_LINE)
    R_main = 18.0
    station_coords = {}

    for i, name in enumerate(MAIN_LINE):
        theta = np.pi / 2 - (2 * np.pi * i / N_main)
        x = R_main * np.cos(theta)
        y = R_main * np.sin(theta)
        station_coords[name] = (x, y, "main")

    x_sungsu, y_sungsu = station_coords["성수"][0], station_coords["성수"][1]
    for i, name in enumerate(SUNGSU_BRANCH):
        station_coords[name] = (x_sungsu, y_sungsu + (i + 1) * 2.5, "sungsu")

    x_sindorim, y_sindorim = station_coords["신도림"][0], station_coords["신도림"][1]
    for i, name in enumerate(SINDORIM_BRANCH):
        station_coords[name] = (x_sindorim - (i + 1) * 2.0, y_sindorim - (i + 1) * 2.0, "sindorim")

    trains = get_realtime_train_positions(seoul_key)
    fig_map = go.Figure()

    main_x = [station_coords[name][0] for name in MAIN_LINE] + [station_coords[MAIN_LINE[0]][0]]
    main_y = [station_coords[name][1] for name in MAIN_LINE] + [station_coords[MAIN_LINE[0]][1]]
    
    fig_map.add_trace(go.Scatter(
        x=main_x, y=main_y,
        mode="lines",
        line=dict(color="rgba(34, 197, 94, 0.15)", width=9),
        hoverinfo="skip",
        showlegend=False
    ))
    fig_map.add_trace(go.Scatter(
        x=main_x, y=main_y,
        mode="lines",
        line=dict(color="#22c55e", width=4),
        hoverinfo="skip",
        showlegend=False
    ))

    # 성수지선 궤도
    branch1_x = [station_coords["성수"][0]] + [station_coords[name][0] for name in SUNGSU_BRANCH]
    branch1_y = [station_coords["성수"][1]] + [station_coords[name][1] for name in SUNGSU_BRANCH]
    
    fig_map.add_trace(go.Scatter(
        x=branch1_x, y=branch1_y,
        mode="lines",
        line=dict(color="rgba(34, 197, 94, 0.15)", width=8, dash="dash"),
        hoverinfo="skip",
        showlegend=False
    ))
    fig_map.add_trace(go.Scatter(
        x=branch1_x, y=branch1_y,
        mode="lines",
        line=dict(color="#16a34a", width=3, dash="dash"),
        hoverinfo="skip",
        showlegend=False
    ))

    # 신도림지선 궤도
    branch2_x = [station_coords["신도림"][0]] + [station_coords[name][0] for name in SINDORIM_BRANCH]
    branch2_y = [station_coords["신도림"][1]] + [station_coords[name][1] for name in SINDORIM_BRANCH]
    
    fig_map.add_trace(go.Scatter(
        x=branch2_x, y=branch2_y,
        mode="lines",
        line=dict(color="rgba(34, 197, 94, 0.15)", width=8, dash="dash"),
        hoverinfo="skip",
        showlegend=False
    ))
    fig_map.add_trace(go.Scatter(
        x=branch2_x, y=branch2_y,
        mode="lines",
        line=dict(color="#16a34a", width=3, dash="dash"),
        hoverinfo="skip",
        showlegend=False
    ))

    # 역 노드 플롯팅
    node_x = [coord[0] for coord in station_coords.values()]
    node_y = [coord[1] for coord in station_coords.values()]
    node_names = list(station_coords.keys())
    
    node_symbols, node_colors, node_sizes, node_border_colors, node_border_widths = [], [], [], [], []
    node_text, node_font_sizes = [], []
    hubs = ["강남", "잠실", "홍대입구", "신도림", "사당", "신림", "시청", "건대입구", "성수", "왕십리", "선릉", "역삼", "교대"]
    
    for name in node_names:
        is_selected = (name == selected_station) or (selected_station.split("(")[0] in name and name.split("(")[0] in selected_station)
        is_hub = any(h in name for h in hubs)
        transfers = TRANSFER_INFO.get(name, [])
        clean_name = name.split("(")[0]
        
        if is_selected:
            node_symbols.append("circle")
            node_colors.append("#0ea5e9")
            node_sizes.append(15)
            node_border_colors.append("#ffffff")
            node_border_widths.append(2.5)
            node_text.append(f"<b><span style='color:#0ea5e9;'>{clean_name}</span></b>")
            node_font_sizes.append(14.0)
        elif transfers and any(transfers):
            node_symbols.append("square")
            node_colors.append("#ffffff")
            node_sizes.append(10)
            node_border_colors.append("#22c55e")
            node_border_widths.append(1.5)
            node_text.append(f"<b>{clean_name}</b>" if is_hub else clean_name)
            node_font_sizes.append(11.5 if is_hub else 10.5)
        else:
            node_symbols.append("circle")
            node_colors.append("#22c55e")
            node_sizes.append(7.0)
            node_border_colors.append("#ffffff")
            node_border_widths.append(1.0)
            node_text.append(f"<b>{clean_name}</b>" if is_hub else clean_name)
            node_font_sizes.append(11.5 if is_hub else 10.5)

    fig_map.add_trace(go.Scatter(
        x=node_x, y=node_y,
        mode="markers+text",
        marker=dict(
            symbol=node_symbols, size=node_sizes, color=node_colors, 
            line=dict(color=node_border_colors, width=node_border_widths)
        ),
        text=node_text,
        textposition="top center",
        textfont=dict(size=node_font_sizes),
        hoverinfo="text",
        hovertext=[f"역명: {name}" + (f"<br>환승 노선: {', '.join(TRANSFER_INFO[name])}" if TRANSFER_INFO.get(name) and any(TRANSFER_INFO[name]) else "") for name in node_names],
        customdata=node_names,
        showlegend=False
    ))

    # 실시간 열차 마킹
    train_x, train_y, train_hover, train_colors, train_symbols = [], [], [], [], []
    stt_map = {"0": "진입 중", "1": "도착 (정차 중)", "2": "출발 (주행 중)"}

    for t in trains:
        station_name = t.get("statnNm", "")
        target_stn = next((s for s in station_coords.keys() if station_name in s), None)
        if target_stn:
            x_c, y_c, line_type = station_coords[target_stn]
            is_up = t.get("updnLine", "0") == "0"
            offset = 0.85 if is_up else -0.85
            
            if line_type == "main":
                dist = np.sqrt(x_c**2 + y_c**2)
                x_val = x_c * (1 + offset/dist)
                y_val = y_c * (1 + offset/dist)
            else:
                x_val = x_c + offset
                y_val = y_c

            train_x.append(x_val)
            train_y.append(y_val)
            direction = "내선순환(상행)" if is_up else "외선순환(하행)"
            status = stt_map.get(t.get("trainSttus", "1"), "정차 중")
            train_hover.append(
                f"🚊 <b>열차번호 {t.get('trainNo')}</b><br>"
                f"━━━━━━━━━━━━━━━━━━<br>"
                f"운행 방향: <span style='color:#3b82f6;'>{direction}</span><br>"
                f"현재 위치: <b>{target_stn}역</b> ({status})<br>"
                f"최종 목적지: <span style='color:#ef4444;'>{t.get('statnTnm', '순환')}행</span>"
            )
            train_colors.append("#3b82f6" if is_up else "#f97316")
            train_symbols.append("triangle-up" if is_up else "triangle-down")

    if train_x:
        fig_map.add_trace(go.Scatter(
            x=train_x, y=train_y,
            mode="markers",
            marker=dict(symbol=train_symbols, size=13, color=train_colors, line=dict(color="#ffffff", width=1.5)),
            hoverinfo="text",
            hovertext=train_hover,
            name="실시간 열차"
        ))

    fig_map.update_layout(
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-28, 28]),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-28, 28]),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=5, r=5, t=5, b=5), height=800,
        legend=dict(
            orientation="h", yanchor="bottom", y=0.02, xanchor="center", x=0.5,
            font=dict(size=11), bgcolor="rgba(255, 255, 255, 0.7)",
            bordercolor="rgba(0,0,0,0.1)", borderwidth=1
        ),
    )
    
    selection = st.plotly_chart(fig_map, use_container_width=True, on_select="rerun")
    
    if selection and "selection" in selection and "points" in selection["selection"]:
        points = selection["selection"]["points"]
        if points and "customdata" in points[0] and points[0]["customdata"]:
            clicked_station = points[0]["customdata"]
            if clicked_station != st.session_state.get("selected_station"):
                st.session_state["selected_station"] = clicked_station
                st.rerun()

    st.markdown("⚠️ 실시간 열차 마커의 삼각형 방향은 **상행/내선순환(▲, 파랑)** 및 **하행/외선순환(▼, 주황)**을 나타내며, 외곽선이 굵은 하늘색 노드는 **현재 설정된 역**을 표기합니다. <b>지도의 역 마커를 직접 클릭하여 관심 역을 즉시 조회할 수 있습니다.</b>", unsafe_allow_html=True)

    st.markdown("---")
    st.subheader(f"⏱️ {selected_station}역 실시간 도착 정보")
    
    with st.spinner(f"{selected_station}역 실시간 도착 정보 조회 중..."):
        arrival_data = get_realtime_station_arrival(selected_station, seoul_key)
        
    if arrival_data:
        arr_cols = st.columns(2)
        up_trains = [a for a in arrival_data if a.get("updnLine") in ["내선", "상행"]]
        down_trains = [a for a in arrival_data if a.get("updnLine") in ["외선", "하행"]]
        
        with arr_cols[0]:
            st.markdown("### 🔵 내선순환 (상행)")
            if up_trains:
                for arr in up_trains:
                    msg = arr.get("arvlMsg2", "정보 없음")
                    train_no = arr.get("btrainNo", "미정")
                    dest_nm = arr.get("trainLineNm", "내선순환")
                    st.markdown(
                        f"""
                        <div style="background:#f1f5f9; padding:10px 15px; border-radius:8px; margin-bottom:8px; border-left:4px solid #3b82f6;">
                            <span style="font-weight:700; color:#1e293b;">{dest_nm}</span> (열차 {train_no}호)<br>
                            <span style="font-size:18px; font-weight:700; color:#3b82f6;">{msg}</span>
                        </div>
                        """, unsafe_allow_html=True
                    )
            else:
                st.info("현재 조회 가능한 내선순환 열차가 없습니다.")
                
        with arr_cols[1]:
            st.markdown("### 🟠 외선순환 (하행)")
            if down_trains:
                for arr in down_trains:
                    msg = arr.get("arvlMsg2", "정보 없음")
                    train_no = arr.get("btrainNo", "미정")
                    dest_nm = arr.get("trainLineNm", "외선순환")
                    st.markdown(
                        f"""
                        <div style="background:#f1f5f9; padding:10px 15px; border-radius:8px; margin-bottom:8px; border-left:4px solid #ef9f27;">
                            <span style="font-weight:700; color:#1e293b;">{dest_nm}</span> (열차 {train_no}호)<br>
                            <span style="font-size:18px; font-weight:700; color:#ef9f27;">{msg}</span>
                        </div>
                        """, unsafe_allow_html=True
                    )
            else:
                st.info("현재 조회 가능한 외선순환 열차가 없습니다.")
    else:
        st.info("실시간 도착 정보가 존재하지 않거나 가져오는데 실패했습니다.")

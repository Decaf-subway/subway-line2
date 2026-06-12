import re
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from pathlib import Path

from core.config import (
    LINE_COLORS,
    SEOUL_ROUTE_STATION_NAMES,
    LINE_SHAPE_POINTS,
    TRANSFER_INFO,
    station_key_from_display,
    line_key_from_station,
    line_sort_key,
)
from services.subway_api import get_realtime_station_arrival, get_realtime_train_positions

BASE_DIR = Path(__file__).resolve().parents[1]
DATASET_PATH = BASE_DIR / "data" / "processed" / "final_dataset_line1_8_230101-241231.csv"


def compact_station_label(station_name):
    return station_name.split("(", 1)[0]


def arrival_time_text(arr):
    if isinstance(arr, dict):
        seconds_text = str(arr.get("barvlDt", "")).strip()
        if seconds_text.isdigit():
            seconds = int(seconds_text)
            if seconds > 0:
                minutes, remain_seconds = divmod(seconds, 60)
                if minutes and remain_seconds:
                    return f"{minutes}분 {remain_seconds}초 후"
                if minutes:
                    return f"{minutes}분 후"
                return f"{remain_seconds}초 후"

        text = str(arr.get("arvlMsg2") or arr.get("arvlMsg3") or "").strip()
    else:
        text = str(arr or "").strip()

    if not text:
        return "정보 없음"

    # [2]번째 전역 (양재시민의숲) 같은 형태를 2번째 전역 전 (양재시민의숲)으로 깔끔하게 포맷팅
    station_match = re.search(r"\[?(\d+)\]?번째\s*전역(?:\s*\((.*?)\))?", text)
    if station_match:
        nth = station_match.group(1)
        st_name = station_match.group(2)
        if st_name:
            return f"{nth}번째 전역 전 ({st_name})"
        return f"{nth}번째 전역 전"

    text = re.sub(r"\s*\d+번째\s*전역.*$", "", text)
    text = re.sub(r"\s*\d+번째전역.*$", "", text)
    return text.strip() or "정보 없음"


def direction_title(selected_line, direction):
    if selected_line == "2호선":
        return "내선순환 (상행)" if direction == "up" else "외선순환 (하행)"
    return "상행" if direction == "up" else "하행"


def render_arrival_card(arr, accent="#3b82f6"):
    msg = arrival_time_text(arr)
    train_line = str(arr.get("trainLineNm") or "").strip()
    train_no = str(arr.get("btrainNo") or "").strip()
    train_info = train_line or "열차 정보"
    if train_no:
        train_info = f"{train_info} (열차 {train_no}호)"

    st.markdown(
        f"""
        <div style="background:#f1f5f9; padding:12px 16px; border-radius:8px; margin-bottom:10px; border-left:4px solid {accent};">
            <span style="font-size:15px; font-weight:700; color:#1e293b;">{train_info}</span><br>
            <span style="font-size:18px; font-weight:700; color:{accent};">{msg}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def interpolate_polyline(points, count):
    if count <= 0:
        return []
    if count == 1:
        return [tuple(points[0])]

    pts = np.asarray(points, dtype=float)
    deltas = np.diff(pts, axis=0)
    seg_lengths = np.sqrt((deltas ** 2).sum(axis=1))
    total_length = float(seg_lengths.sum())
    if total_length == 0:
        return [tuple(pts[0])] * count

    cumulative = np.concatenate([[0.0], np.cumsum(seg_lengths)])
    targets = np.linspace(0, total_length, count)
    coords = []

    for target in targets:
        idx = min(max(np.searchsorted(cumulative, target, side="right") - 1, 0), len(seg_lengths) - 1)
        length = seg_lengths[idx] if seg_lengths[idx] else 1.0
        ratio = (target - cumulative[idx]) / length
        coord = pts[idx] + ratio * (pts[idx + 1] - pts[idx])
        coords.append((float(coord[0]), float(coord[1])))

    return coords


def ellipse_coords(count, center=(0.0, 0.0), radius=(3.6, 2.25), start_angle=2.35):
    if count <= 0:
        return []
    cx, cy = center
    rx, ry = radius
    angles = np.linspace(start_angle, start_angle - 2 * np.pi, count, endpoint=False)
    return [(float(cx + rx * np.cos(angle)), float(cy + ry * np.sin(angle))) for angle in angles]


def station_key_lookup(station_keys):
    return {station_key_from_display(station): station for station in station_keys}


def station_keys_by_names(station_keys, station_names):
    lookup = station_key_lookup(station_keys)
    return [lookup[name] for name in station_names if name in lookup]


def reorder_station_keys(station_keys, station_names):
    ordered = station_keys_by_names(station_keys, station_names)
    used = set(ordered)
    ordered.extend([station for station in station_keys if station not in used])
    return ordered


def build_route_segments(line, station_keys):
    if line == "1호선":
        station_keys = reorder_station_keys(
            station_keys,
            ["서울역", "시청", "종각", "종로3가", "종로5가", "동대문", "동묘앞", "신설동", "제기동", "청량리(서울시립대입구)"],
        )
        coords = [
            (-1.7, -2.2),
            (-1.7, -0.8),
            (-1.45, 0.35),
            (-0.7, 0.95),
            (0.25, 0.95),
            (1.15, 0.95),
            (2.05, 0.95),
            (2.95, 0.95),
            (3.85, 1.12),
            (4.65, 1.55),
        ]
        return [{"name": "1호선 서울 구간", "station_keys": station_keys, "coords": coords[:len(station_keys)], "closed": False}]

    if line == "2호선":
        names = [station_key_from_display(station) for station in station_keys]
        loop_end = names.index("충정로(경기대입구)") + 1 if "충정로(경기대입구)" in names else len(station_keys)
        main_keys = station_keys[:loop_end]
        main_coords = ellipse_coords(len(main_keys))
        coord_map = {station: coord for station, coord in zip(main_keys, main_coords)}

        segments = [{"name": "2호선 본선", "station_keys": main_keys, "coords": main_coords, "closed": True}]

        seongsu_branch = station_keys_by_names(station_keys, ["성수", "용답", "신답", "용두(동대문구청)", "신설동"])
        if len(seongsu_branch) >= 2 and seongsu_branch[0] in coord_map:
            start = coord_map[seongsu_branch[0]]
            segments.append({
                "name": "성수지선",
                "station_keys": seongsu_branch,
                "coords": interpolate_polyline([start, (start[0] + 0.7, start[1] + 0.9), (-0.4, 2.35)], len(seongsu_branch)),
                "closed": False,
            })

        sindorim_branch = station_keys_by_names(station_keys, ["신도림", "도림천", "양천구청", "신정네거리"])
        if len(sindorim_branch) >= 2 and sindorim_branch[0] in coord_map:
            start = coord_map[sindorim_branch[0]]
            segments.append({
                "name": "신도림지선",
                "station_keys": sindorim_branch,
                "coords": interpolate_polyline([start, (start[0] - 0.6, start[1] - 0.7), (start[0] - 1.35, start[1] - 1.45)], len(sindorim_branch)),
                "closed": False,
            })

        return segments

    if line == "5호선":
        lookup = station_key_lookup(station_keys)
        if "강동" in lookup:
            gangdong_idx = station_keys.index(lookup["강동"])
            main_keys = station_keys[:gangdong_idx + 1]
            main_coords = interpolate_polyline([(-5.2, 0.1), (-3.5, 0.15), (-1.5, 0.25), (0.5, 0.1), (2.6, 0.0)], len(main_keys))
            gangdong_coord = main_coords[-1]
            hanam_branch = station_keys_by_names(
                station_keys,
                ["강동", "길동", "굽은다리(강동구민회관앞)", "명일", "고덕", "상일동", "강일", "미사", "하남풍산", "하남시청(덕풍·신장)", "하남검단산"],
            )
            macheon_branch = station_keys_by_names(
                station_keys,
                ["강동", "둔촌동", "올림픽공원(한국체대)", "방이", "오금", "개롱", "거여", "마천"],
            )
            segments = [{"name": "5호선 본선", "station_keys": main_keys, "coords": main_coords, "closed": False}]
            if len(hanam_branch) >= 2:
                segments.append({
                    "name": "상일동·하남 방면",
                    "station_keys": hanam_branch,
                    "coords": interpolate_polyline([gangdong_coord, (3.9, 0.55), (5.4, 1.2)], len(hanam_branch)),
                    "closed": False,
                })
            if len(macheon_branch) >= 2:
                segments.append({
                    "name": "마천 방면",
                    "station_keys": macheon_branch,
                    "coords": interpolate_polyline([gangdong_coord, (3.8, -0.55), (5.0, -1.45)], len(macheon_branch)),
                    "closed": False,
                })
            return segments

    if line == "6호선":
        loop_keys = station_keys_by_names(station_keys, ["응암", "역촌", "불광", "독바위", "연신내", "구산"])
        lookup = station_key_lookup(station_keys)
        if loop_keys and "구산" in lookup:
            after_loop = station_keys[station_keys.index(lookup["구산"]) + 1:]
            main_keys = [loop_keys[0]] + after_loop
            loop_coords = ellipse_coords(len(loop_keys), center=(-3.4, 0.9), radius=(0.8, 0.55), start_angle=np.pi)
            main_coords = interpolate_polyline([loop_coords[0], (-2.4, 0.55), (-0.7, 0.25), (1.1, -0.05), (2.9, -0.1), (4.4, 0.05)], len(main_keys))
            return [
                {"name": "응암순환", "station_keys": loop_keys, "coords": loop_coords, "closed": True},
                {"name": "6호선 본선", "station_keys": main_keys, "coords": main_coords, "closed": False},
            ]

    shape_points = LINE_SHAPE_POINTS.get(line, [(-4.0, 0.0), (-2.0, 0.2), (0.0, 0.0), (2.0, -0.2), (4.0, 0.0)])
    return [{"name": line, "station_keys": station_keys, "coords": interpolate_polyline(shape_points, len(station_keys)), "closed": False}]


def hex_to_rgba(hex_str, alpha=0.15):
    hex_str = hex_str.lstrip('#')
    try:
        r = int(hex_str[0:2], 16)
        g = int(hex_str[2:4], 16)
        b = int(hex_str[4:6], 16)
        return f"rgba({r}, {g}, {b}, {alpha})"
    except Exception:
        return f"rgba(100, 100, 100, {alpha})"


@st.cache_data(show_spinner=False)
def load_route_station_orders():
    if not DATASET_PATH.exists():
        return {}

    df = pd.read_csv(DATASET_PATH, usecols=["역명", "호선"])
    df["역명"] = df["역명"].astype(str)
    df["호선"] = df["호선"].astype(str)

    route_orders = {}
    for line in sorted(df["호선"].dropna().unique(), key=line_sort_key):
        line_df = df[df["호선"] == line][["역명", "호선"]].drop_duplicates()
        station_keys = [f"{row['역명']}_{row['호선']}" for _, row in line_df.iterrows()]
        seoul_station_names = SEOUL_ROUTE_STATION_NAMES.get(line)
        if seoul_station_names:
            lookup = {station_key_from_display(station): station for station in station_keys}
            route_orders[line] = [lookup[name] for name in seoul_station_names if name in lookup]
        else:
            route_orders[line] = station_keys
    return route_orders


def get_transfer_station_names(route_orders):
    station_lines = {}
    for line, station_keys in route_orders.items():
        for station_key in station_keys:
            station_lines.setdefault(station_key_from_display(station_key), set()).add(line)
    return {name for name, lines in station_lines.items() if len(lines) > 1}


def render_route_map(route_orders, selected_station, selected_line, LINE_OPTIONS, seoul_key=None):
    if selected_line not in route_orders:
        st.warning("노선도에 사용할 역 순서 데이터를 찾지 못했습니다.")
        return

    transfer_names = get_transfer_station_names(route_orders)
    station_keys = route_orders[selected_line]
    segments = build_route_segments(selected_line, station_keys)
    color = LINE_COLORS.get(selected_line, "#64748b")
    fig = go.Figure()

    # 1. 궤도선(Lines)을 먼저 전부 추가 (Glow 효과 + 실선)
    for segment in segments:
        coords = segment["coords"]
        if not coords:
            continue
        x_values = [coord[0] for coord in coords]
        y_values = [coord[1] for coord in coords]
        line_x = x_values + ([x_values[0]] if segment.get("closed") else [])
        line_y = y_values + ([y_values[0]] if segment.get("closed") else [])

        # Glow 효과 배경선
        fig.add_trace(go.Scatter(
            x=line_x,
            y=line_y,
            mode="lines",
            line=dict(color=hex_to_rgba(color, 0.15), width=12),
            hoverinfo="skip",
            showlegend=False,
        ))
        # 메인 실선
        fig.add_trace(go.Scatter(
            x=line_x,
            y=line_y,
            mode="lines",
            line=dict(color=color, width=4),
            hoverinfo="skip",
            showlegend=False,
        ))

    # 2. 역 노드(Markers) 추가
    for segment in segments:
        segment_keys = segment["station_keys"]
        coords = segment["coords"]
        if not segment_keys or not coords:
            continue
        x_values = [coord[0] for coord in coords]
        y_values = [coord[1] for coord in coords]
        
        hover_labels = []
        node_symbols = []
        node_colors = []
        node_sizes = []
        node_border_colors = []
        node_border_widths = []
        node_text = []
        node_font_sizes = []

        for idx, station_key in enumerate(segment_keys):
            station_name = station_key_from_display(station_key)
            is_selected = station_key == selected_station
            is_transfer = station_name in transfer_names
            hover_labels.append(f"{station_name} ({selected_line})")
            clean_name = compact_station_label(station_name)

            if is_selected:
                node_symbols.append("circle")
                node_colors.append("#0ea5e9")  # 하늘색 포커스
                node_sizes.append(15)
                node_border_colors.append("#ffffff")
                node_border_widths.append(2.5)
                node_text.append(f"<b><span style='color:#0ea5e9;'>{clean_name}</span></b>")
                node_font_sizes.append(14.5)
            elif is_transfer:
                node_symbols.append("square")
                node_colors.append("#ffffff")  # 환승 허브 스퀘어
                node_sizes.append(10)
                node_border_colors.append(color)
                node_border_widths.append(1.5)
                node_text.append(f"<b>{clean_name}</b>")
                node_font_sizes.append(12.0)
            else:
                node_symbols.append("circle")
                node_colors.append(color)
                node_sizes.append(7.0)
                node_border_colors.append("#ffffff")
                node_border_widths.append(1.0)
                node_text.append(clean_name)
                node_font_sizes.append(11.0)

        fig.add_trace(go.Scatter(
            x=x_values,
            y=y_values,
            mode="markers+text",
            marker=dict(
                symbol=node_symbols,
                size=node_sizes,
                color=node_colors,
                line=dict(color=node_border_colors, width=node_border_widths)
            ),
            text=node_text,
            textposition="top center",
            textfont=dict(size=node_font_sizes),
            customdata=segment_keys,
            hoverinfo="text",
            hovertext=hover_labels,
            showlegend=False,
        ))

    # 3. 실시간 열차 마커 오버레이
    train_positions = get_realtime_train_positions(seoul_key or "sample", selected_line) if seoul_key else []
    if train_positions:
        station_coord_map = {}
        for segment in segments:
            for station_key, coord in zip(segment["station_keys"], segment["coords"]):
                station_coord_map.setdefault(station_key_from_display(station_key), coord)

        stt_map = {"0": "진입중", "1": "상행(운행중)", "2": "하행(운행중)"}
        train_x, train_y, train_hover, train_colors, train_symbols = [], [], [], [], []

        for train in train_positions:
            station_name = train.get("statnNm", "")
            target_station = next((name for name in station_coord_map if station_name in name or name in station_name), None)
            if not target_station:
                continue
            x_c, y_c = station_coord_map[target_station]
            is_up = train.get("updnLine", "0") == "0"
            offset = 0.25 if is_up else -0.25  # 1-8호선 맵 스케일 조절 오프셋
            train_x.append(x_c + offset)
            train_y.append(y_c)
            train_colors.append("#3b82f6" if is_up else "#f97316")
            train_symbols.append("triangle-up" if is_up else "triangle-down")
            train_hover.append(
                f"🚊 <b>열차번호 {train.get('trainNo', '미상')}</b><br>"
                f"━━━━━━━━━━━━━━━━━━<br>"
                f"현재 위치: <b>{target_station}역</b> ({stt_map.get(train.get('trainSttus', '1'), '운행중')})<br>"
                f"최종 목적지: <span style='color:#ef4444;'>{train.get('statnTnm', '미상')}행</span>"
            )

        if train_x:
            fig.add_trace(go.Scatter(
                x=train_x,
                y=train_y,
                mode="markers",
                marker=dict(symbol=train_symbols, size=13, color=train_colors, line=dict(color="#ffffff", width=1.5)),
                hoverinfo="text",
                hovertext=train_hover,
                name="실시간 열차",
                showlegend=False,
            ))

    # 4. 레이아웃 업데이트 및 출력
    fig.update_layout(
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-6, 6]),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-4.5, 4.5]),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=5, r=5, t=5, b=5),
        height=1000,
    )
    
    # 5. plotly_chart 호출 및 클릭 연동
    selection = st.plotly_chart(fig, use_container_width=True, on_select="rerun")
    
    if selection and "selection" in selection and "points" in selection["selection"]:
        points = selection["selection"]["points"]
        if points and "customdata" in points[0] and points[0]["customdata"]:
            clicked_station = points[0]["customdata"]
            if clicked_station != st.session_state.get("selected_station"):
                st.session_state["selected_station"] = clicked_station
                clicked_line = line_key_from_station(clicked_station)
                if clicked_line and clicked_line in LINE_OPTIONS:
                    st.session_state["selected_line"] = clicked_line
                st.rerun()

    st.markdown("⚠️ 실시간 열차 마커의 삼각형 방향은 **상행(▲, 파랑)** 및 **하행(▼, 주황)**을 나타내며, 외곽선이 굵은 하늘색 노드는 **현재 설정된 역**을 표기합니다. <b>지도의 역 마커를 직접 클릭하여 관심 역을 즉시 조회할 수 있습니다.</b>", unsafe_allow_html=True)

    ordered_station_keys = []
    seen_stations = set()
    for segment in segments:
        for station in segment["station_keys"]:
            if station not in seen_stations:
                ordered_station_keys.append(station)
                seen_stations.add(station)

    if ordered_station_keys:
        line_station_names = [station_key_from_display(station) for station in ordered_station_keys]
        st.markdown(f"**{selected_line} 전체 역 이름**")
        st.write(" · ".join(line_station_names))

    st.markdown("---")
    arrival_station_name = station_key_from_display(selected_station) if selected_station else selected_line
    st.subheader(f"⏱️ {arrival_station_name}역 실시간 도착 정보")
    with st.spinner("실시간 도착 정보를 조회하는 중..."):
        arrival_data = get_realtime_station_arrival(arrival_station_name, seoul_key or "sample") if selected_station else []

    if arrival_data:
        arr_cols = st.columns(2)
        up_trains = [a for a in arrival_data if a.get("updnLine") in ["상행", "내선"] or a.get("updnLine") == "0"]
        down_trains = [a for a in arrival_data if a.get("updnLine") in ["하행", "외선"] or a.get("updnLine") == "1"]

        with arr_cols[0]:
            st.markdown(f"### 🔵 {direction_title(selected_line, 'up')}")
            if up_trains:
                for arr in up_trains[:2]:
                    render_arrival_card(arr, "#3b82f6")
            else:
                st.info("현재 조회 가능한 상행 열차 정보가 없습니다.")

        with arr_cols[1]:
            st.markdown(f"### 🟠 {direction_title(selected_line, 'down')}")
            if down_trains:
                for arr in down_trains[:2]:
                    render_arrival_card(arr, "#ef9f27")
            else:
                st.info("현재 조회 가능한 하행 열차 정보가 없습니다.")
    else:
        st.info("실시간 도착 정보가 없거나 조회에 실패했습니다.")


def render_map_tab(selected_station, selected_line, LINE_OPTIONS, seoul_key=None):
    route_orders = load_route_station_orders()
    render_route_map(route_orders, selected_station, selected_line, LINE_OPTIONS, seoul_key)

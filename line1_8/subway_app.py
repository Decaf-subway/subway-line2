"""
서울 1-8호선 실시간 혼잡도 예측 대시보드
실행: streamlit run subway_app.py
"""

import json
import os
import sys
from datetime import date, datetime
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from components.styles import inject_custom_styles
from core.model_loader import load_all_models, load_lstm_base_dataset
from core.predictor import get_congestion, get_station_avg, predict, predict_day
from services.subway_api import check_holiday
from services.weather_api import get_weather_with_fallback


DATASET_PATH = BASE_DIR / "data" / "processed" / "final_dataset_line1_8_230101-241231.csv"
LINE_COLORS = {
    "1호선": "#0052A4",
    "2호선": "#00A84D",
    "3호선": "#EF7C1C",
    "4호선": "#00A5DE",
    "5호선": "#996CAC",
    "6호선": "#CD7C2F",
    "7호선": "#747F00",
    "8호선": "#E6186C",
}
SEOUL_ROUTE_STATION_NAMES = {
    "1호선": [
        "서울역",
        "시청",
        "종각",
        "종로3가",
        "종로5가",
        "동대문",
        "동묘앞",
        "신설동",
        "제기동",
        "청량리(서울시립대입구)",
    ],
    "2호선": [
        "시청",
        "을지로입구",
        "을지로3가",
        "을지로4가",
        "동대문역사문화공원(DDP)",
        "신당",
        "상왕십리",
        "왕십리(성동구청)",
        "한양대",
        "뚝섬",
        "성수",
        "건대입구",
        "구의(광진구청)",
        "강변(동서울터미널)",
        "잠실나루",
        "잠실(송파구청)",
        "잠실새내",
        "종합운동장",
        "삼성(무역센터)",
        "선릉",
        "역삼",
        "강남",
        "교대(법원.검찰청)",
        "서초",
        "방배",
        "사당",
        "낙성대(강감찬)",
        "서울대입구(관악구청)",
        "봉천",
        "신림",
        "신대방",
        "구로디지털단지",
        "대림(구로구청)",
        "신도림",
        "문래",
        "영등포구청",
        "당산",
        "합정",
        "홍대입구",
        "신촌",
        "이대",
        "아현",
        "충정로(경기대입구)",
        "용답",
        "신답",
        "신설동",
        "도림천",
        "양천구청",
        "신정네거리",
        "용두(동대문구청)",
    ],
    "3호선": [
        "구파발",
        "연신내",
        "불광",
        "녹번",
        "홍제",
        "무악재",
        "독립문",
        "경복궁(정부서울청사)",
        "안국",
        "종로3가",
        "을지로3가",
        "충무로",
        "동대입구",
        "약수",
        "금호",
        "옥수",
        "압구정",
        "신사",
        "잠원",
        "고속터미널",
        "교대(법원.검찰청)",
        "남부터미널(예술의전당)",
        "양재(서초구청)",
        "매봉",
        "도곡",
        "대치",
        "학여울",
        "대청",
        "일원",
        "수서",
        "가락시장",
        "경찰병원",
        "오금",
    ],
    "4호선": [
        "당고개",
        "상계",
        "노원",
        "창동",
        "쌍문",
        "수유(강북구청)",
        "미아(서울사이버대학)",
        "미아사거리",
        "길음",
        "성신여대입구(돈암)",
        "한성대입구(삼선교)",
        "혜화",
        "동대문",
        "동대문역사문화공원(DDP)",
        "충무로",
        "명동",
        "회현(남대문시장)",
        "서울역",
        "숙대입구(갈월)",
        "삼각지",
        "신용산",
        "이촌(국립중앙박물관)",
        "동작(현충원)",
        "총신대입구(이수)",
        "사당",
        "남태령",
    ],
    "5호선": [
        "방화",
        "개화산",
        "김포공항",
        "송정",
        "마곡",
        "발산",
        "우장산",
        "화곡",
        "까치산",
        "신정(은행정)",
        "목동",
        "오목교(목동운동장앞)",
        "양평",
        "영등포구청",
        "영등포시장",
        "신길",
        "여의도",
        "여의나루",
        "마포",
        "공덕",
        "애오개",
        "충정로(경기대입구)",
        "서대문",
        "광화문(세종문화회관)",
        "종로3가",
        "을지로4가",
        "동대문역사문화공원(DDP)",
        "청구",
        "신금호",
        "행당",
        "왕십리(성동구청)",
        "마장",
        "답십리",
        "장한평",
        "군자(능동)",
        "아차산(어린이대공원후문)",
        "광나루(장신대)",
        "천호(풍납토성)",
        "강동",
        "길동",
        "굽은다리(강동구민회관앞)",
        "명일",
        "고덕",
        "상일동",
        "강일",
        "둔촌동",
        "올림픽공원(한국체대)",
        "방이",
        "오금",
        "개롱",
        "거여",
        "마천",
    ],
    "6호선": [
        "응암",
        "역촌",
        "불광",
        "독바위",
        "연신내",
        "구산",
        "새절(신사)",
        "증산(명지대앞)",
        "디지털미디어시티",
        "월드컵경기장(성산)",
        "마포구청",
        "망원",
        "합정",
        "상수",
        "광흥창(서강)",
        "대흥(서강대앞)",
        "공덕",
        "효창공원앞",
        "삼각지",
        "녹사평(용산구청)",
        "이태원",
        "한강진",
        "버티고개",
        "약수",
        "청구",
        "신당",
        "동묘앞",
        "창신",
        "보문",
        "안암(고대병원앞)",
        "고려대(종암)",
        "월곡(동덕여대)",
        "상월곡(한국과학기술연구원)",
        "돌곶이",
        "석계",
        "태릉입구",
        "화랑대(서울여대입구)",
        "봉화산(서울의료원)",
        "신내",
    ],
    "7호선": [
        "도봉산",
        "수락산",
        "마들",
        "노원",
        "중계",
        "하계",
        "공릉(서울과학기술대)",
        "태릉입구",
        "먹골",
        "중화",
        "상봉(시외버스터미널)",
        "면목",
        "사가정",
        "용마산(용마폭포공원)",
        "중곡",
        "군자(능동)",
        "어린이대공원(세종대)",
        "건대입구",
        "뚝섬유원지",
        "청담",
        "강남구청",
        "학동",
        "논현",
        "반포",
        "고속터미널",
        "내방",
        "이수",
        "남성",
        "숭실대입구(살피재)",
        "상도",
        "장승배기",
        "신대방삼거리",
        "보라매",
        "신풍",
        "대림(구로구청)",
        "남구로",
        "가산디지털단지",
        "천왕",
        "온수(성공회대입구)",
    ],
    "8호선": [
        "암사역사공원",
        "암사",
        "천호(풍납토성)",
        "강동구청",
        "몽촌토성(평화의문)",
        "잠실(송파구청)",
        "석촌",
        "송파",
        "가락시장",
        "문정",
        "장지",
        "복정",
    ],
}


def load_env_file():
    env_path = BASE_DIR / ".env"
    if env_path.exists():
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ[k.strip()] = v.strip()


def format_station_label(station_key: str) -> str:
    if "_" not in station_key:
        return station_key
    base, line = station_key.rsplit("_", 1)
    return f"{base} ({line})"


def station_key_from_display(station_key: str) -> str:
    return station_key.rsplit("_", 1)[0]


def line_key_from_station(station_key: str) -> str:
    if "_" not in station_key:
        return ""
    return station_key.rsplit("_", 1)[1]


def line_sort_key(line_key: str) -> int:
    digits = "".join(ch for ch in line_key if ch.isdigit())
    return int(digits) if digits else 999


def stations_for_line(stations, line_key: str):
    return sorted(
        [station for station in stations if line_key_from_station(station) == line_key],
        key=station_key_from_display,
    )


def find_station_by_name(stations, station_name: str, line_key: str | None = None):
    for station in stations:
        if station_key_from_display(station) == station_name and (line_key is None or line_key_from_station(station) == line_key):
            return station
    return None


def select_line_station(label_prefix, stations, line_options, default_station, key_prefix):
    default_line = line_key_from_station(default_station) if default_station in stations else line_options[0]
    if default_line not in line_options:
        default_line = line_options[0]

    line_state_key = f"{key_prefix}_line"
    station_state_key = f"{key_prefix}_station"

    if line_state_key not in st.session_state or st.session_state[line_state_key] not in line_options:
        st.session_state[line_state_key] = default_line

    selected_line = st.selectbox(
        f"{label_prefix} 호선",
        line_options,
        index=line_options.index(st.session_state[line_state_key]),
        key=line_state_key,
    )

    line_stations = stations_for_line(stations, selected_line)
    if station_state_key not in st.session_state or st.session_state[station_state_key] not in line_stations:
        preferred_station = default_station if line_key_from_station(default_station) == selected_line else None
        st.session_state[station_state_key] = preferred_station if preferred_station in line_stations else line_stations[0]

    return st.selectbox(
        f"{label_prefix} 역",
        line_stations,
        index=line_stations.index(st.session_state[station_state_key]),
        format_func=station_key_from_display,
        key=station_state_key,
    )


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


LINE_SHAPE_POINTS = {
    "3호선": [(-3.4, 2.7), (-2.6, 1.9), (-1.5, 1.2), (-0.6, 0.4), (0.2, -0.2), (0.9, -0.8), (1.6, -1.5), (2.5, -2.5)],
    "4호선": [(-1.4, 3.0), (-1.0, 2.1), (-0.5, 1.2), (0.0, 0.3), (0.0, -0.5), (-0.5, -1.2), (-1.0, -1.9), (-1.8, -2.8)],
    "7호선": [(-1.9, 3.0), (-1.4, 2.2), (-0.9, 1.4), (-0.1, 0.6), (0.9, 0.2), (1.7, -0.3), (0.9, -1.0), (-0.1, -1.5), (-1.5, -2.0), (-3.0, -2.4)],
    "8호선": [(-0.4, 2.7), (-0.3, 1.7), (-0.1, 0.8), (0.1, -0.1), (0.4, -1.0), (0.7, -2.0), (1.0, -2.8)],
}


def compact_station_label(station_name):
    return station_name.split("(", 1)[0]


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


def render_route_map(route_orders, selected_station, selected_line):
    if selected_line not in route_orders:
        st.warning("노선도에 사용할 역 순서 데이터를 찾지 못했습니다.")
        return

    transfer_names = get_transfer_station_names(route_orders)
    station_keys = route_orders[selected_line]
    segments = build_route_segments(selected_line, station_keys)
    color = LINE_COLORS.get(selected_line, "#64748b")
    fig = go.Figure()

    for segment in segments:
        segment_keys = segment["station_keys"]
        coords = segment["coords"]
        if not segment_keys or not coords:
            continue

        x_values = [coord[0] for coord in coords]
        y_values = [coord[1] for coord in coords]
        line_x = x_values + ([x_values[0]] if segment.get("closed") else [])
        line_y = y_values + ([y_values[0]] if segment.get("closed") else [])
        marker_sizes = []
        station_labels = []
        hover_labels = []

        for idx, station_key in enumerate(segment_keys):
            station_name = station_key_from_display(station_key)
            is_selected = station_key == selected_station
            is_transfer = station_name in transfer_names
            marker_sizes.append(16 if is_selected else 9 if is_transfer else 6)
            station_labels.append(compact_station_label(station_name) if is_selected or is_transfer or idx in (0, len(segment_keys) - 1) else "")
            hover_labels.append(f"{station_name} ({selected_line})")

        fig.add_trace(go.Scatter(
            x=line_x,
            y=line_y,
            mode="lines",
            line=dict(color=color, width=6),
            hoverinfo="skip",
            showlegend=False,
        ))
        fig.add_trace(go.Scatter(
            x=x_values,
            y=y_values,
            mode="markers+text",
            marker=dict(
                size=marker_sizes,
                color=["#ffffff" if station_key == selected_station else color for station_key in segment_keys],
                line=dict(color=color, width=[4 if station_key == selected_station else 2 for station_key in segment_keys]),
            ),
            text=station_labels,
            textposition="top center",
            textfont=dict(size=11, color="#111827"),
            customdata=hover_labels,
            hovertemplate="%{customdata}<extra></extra>",
            showlegend=False,
        ))

    fig.update_layout(
        title=f"서울 {selected_line} 노선도",
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        plot_bgcolor="white",
        paper_bgcolor="white",
        height=520,
        margin=dict(l=24, r=16, t=56, b=12),
    )
    st.plotly_chart(fig, use_container_width=True)

    ordered_station_keys = []
    seen_stations = set()
    for segment in segments:
        for station in segment["station_keys"]:
            if station not in seen_stations:
                ordered_station_keys.append(station)
                seen_stations.add(station)

    if ordered_station_keys:
        line_station_names = [station_key_from_display(station) for station in ordered_station_keys]
        st.markdown(f"**{selected_line} 역 순서**")
        st.caption(" · ".join(line_station_names))


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


def load_transfer_info():
    path = BASE_DIR / "data" / "transfer_info.json"
    if not path.exists():
        return {}

    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


load_env_file()

st.set_page_config(
    page_title="서울 1-8호선 혼잡도 예측",
    page_icon="🚇",
    layout="wide",
    initial_sidebar_state="expanded",
)

TRANSFER_INFO = load_transfer_info()

inject_custom_styles()

LSTM_BASE_DF = load_lstm_base_dataset()
ALL_MODELS, le_station, model_loaded = load_all_models()

if not model_loaded:
    st.error("사용 가능한 예측 모델을 찾지 못했습니다. `models/` 경로를 확인해 주세요.")
    st.stop()

STATIONS = sorted(list(le_station.classes_)) if le_station is not None else []

if not STATIONS:
    st.error("선택할 수 있는 역 목록을 불러오지 못했습니다.")
    st.stop()

LINE_OPTIONS = sorted(
    {line_key_from_station(station) for station in STATIONS if line_key_from_station(station)},
    key=line_sort_key,
)

if not LINE_OPTIONS:
    st.error("선택할 수 있는 호선 목록을 불러오지 못했습니다.")
    st.stop()

if "selected_station" not in st.session_state or st.session_state["selected_station"] not in STATIONS:
    st.session_state["selected_station"] = "강남_2호선" if "강남_2호선" in STATIONS else STATIONS[0]

if "selected_line" not in st.session_state or st.session_state["selected_line"] not in LINE_OPTIONS:
    current_line = line_key_from_station(st.session_state["selected_station"])
    st.session_state["selected_line"] = current_line if current_line in LINE_OPTIONS else LINE_OPTIONS[0]

if "selected_model" not in st.session_state:
    st.session_state["selected_model"] = "LightGBM"

st.title("서울 1-8호선 실시간 혼잡도 예측")
st.caption("1-8호선 역별 승차·하차 인원과 날씨를 함께 반영해 혼잡도를 예측합니다.")

loaded_models = [m_name for m_name, info in ALL_MODELS.items() if info.get("loaded", False)]

with st.sidebar:
    st.header("예측 설정")

    active_model = st.selectbox(
        "사용 모델",
        loaded_models,
        index=loaded_models.index(st.session_state["selected_model"]) if st.session_state["selected_model"] in loaded_models else 0,
    )
    st.session_state["selected_model"] = active_model

    selected_line = st.selectbox(
        "호선 선택",
        LINE_OPTIONS,
        index=LINE_OPTIONS.index(st.session_state["selected_line"]),
    )
    st.session_state["selected_line"] = selected_line

    line_stations = stations_for_line(STATIONS, selected_line)
    if st.session_state["selected_station"] not in line_stations:
        preferred_station = f"강남_{selected_line}"
        st.session_state["selected_station"] = preferred_station if preferred_station in line_stations else line_stations[0]

    selected_station = st.selectbox(
        "역 선택",
        line_stations,
        index=line_stations.index(st.session_state["selected_station"]),
        format_func=station_key_from_display,
    )
    st.session_state["selected_station"] = selected_station

    transfer_name = station_key_from_display(selected_station)
    transfers = TRANSFER_INFO.get(transfer_name, [])
    if transfers and any(transfers):
        st.caption(f"환승: {', '.join([t for t in transfers if t])}")

    st.divider()

    selected_date = st.date_input(
        "날짜",
        value=date.today(),
        min_value=date(2020, 1, 1),
        max_value=date(2030, 12, 31),
    )
    selected_hour = st.slider("시간", min_value=5, max_value=23, value=9, format="%d시")

    st.divider()
    st.subheader("날씨")
    weather_mode = st.radio(
        "입력 방식",
        ["기상청 API (자동)", "수동 입력"],
        index=0,
    )

if weather_mode == "기상청 API (자동)":
    api_key = os.environ.get("KMA_API_KEY", "")
    cache_key = (selected_date.isoformat(), selected_hour)
    if (
        "kma_weather" not in st.session_state
        or st.session_state.get("weather_cache_key") != cache_key
    ):
        with st.spinner("기상청 데이터를 불러오는 중..."):
            result = get_weather_with_fallback(api_key, selected_date, selected_hour)
            st.session_state["kma_weather"] = result[:3]
            st.session_state["weather_source"] = result[3]
            st.session_state["weather_cache_key"] = cache_key

    temp, rain, snow = st.session_state["kma_weather"]
    source = st.session_state.get("weather_source", "기본값")

    days_diff = (selected_date - date.today()).days
    if days_diff < 0 or days_diff > 10:
        st.warning("기상청 자동 입력은 오늘 기준 10일 이내 날짜에 더 잘 맞습니다. 그 밖의 날짜는 수동 입력을 권장합니다.")
    else:
        st.success(f"현재 날씨 ({source}): 기온 {temp}℃ / 강수량 {rain}mm / 적설 {snow}mm")

    if st.button("날씨 새로고침"):
        st.session_state.pop("kma_weather", None)
        st.rerun()
else:
    temp = st.slider("기온 (℃)", min_value=-15.0, max_value=40.0, value=15.0, step=0.5)
    rain = st.slider("강수량 (mm)", min_value=0.0, max_value=100.0, value=0.0, step=0.5)
    snow = st.slider("적설 (mm)", min_value=0.0, max_value=50.0, value=0.0, step=0.5)

dt = datetime.combine(selected_date, datetime.min.time())
hol_name, hol_msg = check_holiday(dt)
if hol_msg:
    icon = "🎌" if hol_name else "📅"
    st.markdown(f'<div class="holiday-banner">{icon} {hol_msg}</div>', unsafe_allow_html=True)

base_avg = get_station_avg(selected_station, LSTM_BASE_DF)
ROUTE_ORDERS = load_route_station_orders()

tab0, tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "노선도",
    "시간대 예측",
    "날씨 비교",
    "역별 랭킹",
    "최적 탑승 시간",
    "모델별 예측치 비교",
])

with tab0:
    default_route_line = "1호선" if "1호선" in LINE_OPTIONS else LINE_OPTIONS[0]
    if "route_map_line" not in st.session_state or st.session_state["route_map_line"] not in LINE_OPTIONS:
        st.session_state["route_map_line"] = default_route_line

    route_line = st.selectbox(
        "노선도 호선",
        LINE_OPTIONS,
        index=LINE_OPTIONS.index(st.session_state["route_map_line"]),
        key="route_map_line",
    )
    route_selected_station = selected_station if line_key_from_station(selected_station) == route_line else ""
    render_route_map(ROUTE_ORDERS, route_selected_station, route_line)

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
        cong, ccol, cbg = get_congestion(b, selected_station)
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

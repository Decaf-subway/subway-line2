import os
import json
import streamlit as st
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]

# ── 1-8호선 색상 정의 ─────────────────────────────────────────────────────────────
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

# ── 1-8호선 주요 노선도 역 순서 정의 ────────────────────────────────────────────────
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

# ── 노선도 노선 궤도 점 곡선 제어점 ──────────────────────────────────────────────
LINE_SHAPE_POINTS = {
    "3호선": [(-3.4, 2.7), (-2.6, 1.9), (-1.5, 1.2), (-0.6, 0.4), (0.2, -0.2), (0.9, -0.8), (1.6, -1.5), (2.5, -2.5)],
    "4호선": [(-1.4, 3.0), (-1.0, 2.1), (-0.5, 1.2), (0.0, 0.3), (0.0, -0.5), (-0.5, -1.2), (-1.0, -1.9), (-1.8, -2.8)],
    "7호선": [(-1.9, 3.0), (-1.4, 2.2), (-0.9, 1.4), (-0.1, 0.6), (0.9, 0.2), (1.7, -0.3), (0.9, -1.0), (-0.1, -1.5), (-1.5, -2.0), (-3.0, -2.4)],
    "8호선": [(-0.4, 2.7), (-0.3, 1.7), (-0.1, 0.8), (0.1, -0.1), (0.4, -1.0), (0.7, -2.0), (1.0, -2.8)],
}

# ── 지하철 호선별 정적 환승역 정보 로딩 (노선도 맵핑 및 팝업용) ──────────────────────────
TRANSFER_INFO = {}
if (BASE_DIR / "data" / "transfer_info.json").exists():
    try:
        with open(BASE_DIR / "data" / "transfer_info.json", "r", encoding="utf-8") as f:
            TRANSFER_INFO = json.load(f)
    except Exception:
        pass

# ── 기상청 과거/실시간 날씨 데이터를 가져오기 위한 디폴트 서비스 인증키 ─────────────────
DEFAULT_KMA_KEY = "c8a63dac44b7ee08af43d9426d891501328ed2e638e432fe873ff9bb28d11484"

# ── 모델 학습 시 사용한 피처 컬럼 정의 ──────────────────────────────────────────────
FEATURE_COLS = [
    "역명_enc",
    "시간", "시간_sin", "시간_cos",
    "요일", "요일_sin", "요일_cos",
    "월",   "월_sin",   "월_cos",
    "공휴일여부", "비근무일",
    "출근피크", "퇴근피크",
    "기온", "강수량", "적설",
    "강수_여부", "적설_여부", "불쾌지수",
    "공휴일_유형", "연휴_여부", "공휴일_전날", "공휴일_다음날",
]

# ── 1-8호선 전용 헬퍼 함수들 ─────────────────────────────────────────────────────
def format_station_label(station_key: str) -> str:
    if "_" not in station_key:
        return station_key
    base, line = station_key.rsplit("_", 1)
    return f"{base} ({line})"

def station_key_from_display(station_key: str) -> str:
    if "_" not in station_key:
        return station_key
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

# 각 역별 하루 평균 혼잡 기준 인원 (기본 폴백용 빈 딕셔너리)
STATION_AVG = {}


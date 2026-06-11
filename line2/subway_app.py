"""
서울 지하철 2호선 실시간 관제 및 승하차 예측 앱
===============================================
실행: streamlit run subway_app.py
"""

import os
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))
# TensorFlow의 macOS Metal 플러그인 초기화 지연 방지 (CPU 모드 강제)
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import streamlit as st
from datetime import date, datetime

# ── 1. 페이지 기본 설정 ──────────────────────────────────────────────────────────
st.set_page_config(
    page_title="2호선 실시간 관제 및 승하차 예측",
    page_icon="🚇",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── 2. 커스텀 CSS 스타일 인젝션 ──────────────────────────────────────────────────
from components.styles import inject_custom_styles
inject_custom_styles()

# ── 3. 환경변수 로딩 ─────────────────────────────────────────────────────────────
env_path = BASE_DIR / ".env"
if env_path.exists():
    with open(env_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ[k.strip()] = v.strip()

# ── 4. AI 모델 및 데이터셋 로딩 ──────────────────────────────────────────────────
from core.config import ALL_LINE2_STATIONS, DEFAULT_KMA_KEY, TRANSFER_INFO
from core.model_loader import load_all_models, load_lstm_base_dataset

LSTM_BASE_DF = load_lstm_base_dataset()
ALL_MODELS, le_station, model_loaded = load_all_models()
model_loaded_state = any(m.get("loaded", False) for m in ALL_MODELS.values())

if model_loaded:
    STATIONS = sorted(list(le_station.classes_))
else:
    STATIONS = sorted(ALL_LINE2_STATIONS)

# ── 5. UI 및 레이아웃 정의 ───────────────────────────────────────────────────────
st.title("🚇 서울 지하철 2호선 실시간 관제 및 승하차 예측")
if not model_loaded_state:
    st.error("❌ 필수 AI 모델 파일을 전혀 찾을 수 없습니다. `models/` 디렉토리에 학습 완료된 모델 파일을 안전하게 배치해 주세요.")
    st.stop()

# ── 6. 사이드바 설정 영역 ────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ 예측 설정")

    loaded_models = [m_name for m_name, info in ALL_MODELS.items() if info.get("loaded", False)]
    if not loaded_models:
        loaded_models = ["XGBoost"]
        
    active_model = st.selectbox(
        "🔮 활성 예측 AI 모델",
        loaded_models,
        index=0,
        help="메인 관제 및 예측 탭에서 연산에 사용할 기준 모델을 지정합니다."
    )
    st.success(f"✅ {active_model} 모델 작동 중")
    st.divider()

    if "selected_station" not in st.session_state:
        st.session_state["selected_station"] = "강남" if "강남" in STATIONS else STATIONS[0]
        
    selected_station = st.selectbox(
        "역을 선택하세요", 
        STATIONS, 
        index=STATIONS.index(st.session_state["selected_station"]) if st.session_state["selected_station"] in STATIONS else 0
    )
    st.session_state["selected_station"] = selected_station

    transfers = TRANSFER_INFO.get(selected_station, [])
    if transfers and any(transfers):
        st.caption(f"환승: {' · '.join([t for t in transfers if t])}")
    st.divider()

    selected_date = st.date_input("날짜", value=date.today(), min_value=date(2020,1,1), max_value=date(2030,12,31))
    selected_hour = st.slider("시간", min_value=5, max_value=23, value=9, format="%d시")
    st.divider()

    st.subheader("🌤 날씨 설정")
    weather_mode = st.radio("날씨 입력 방식", ["기상청 API (실시간 자동)", "수동 입력"], index=0)

# ── 7. 날씨 및 공휴일 데이터 연동 ────────────────────────────────────────────────
from services.weather_api import get_weather_with_fallback
from services.subway_api import check_holiday

if weather_mode == "기상청 API (실시간 자동)":
    api_key = os.environ.get("KMA_API_KEY", DEFAULT_KMA_KEY)
    if "kma_weather" not in st.session_state or st.session_state.get("last_date") != selected_date or st.session_state.get("last_hour") != selected_hour:
        with st.spinner("기상청 예보 데이터 실시간 동기화 중..."):
            result = get_weather_with_fallback(api_key, selected_date, selected_hour)
            st.session_state["kma_weather"] = result[:3]
            st.session_state["weather_source"] = result[3]
            st.session_state["last_date"] = selected_date
            st.session_state["last_hour"] = selected_hour

    temp, rain, snow = st.session_state["kma_weather"]
    source = st.session_state.get("weather_source", "기본값")
    
    days_diff = (selected_date - date.today()).days
    if days_diff < 0 or days_diff > 10:
        st.warning("⚠️ 실시간 기상청 예보는 **오늘부터 10일 이내**의 날짜만 지원합니다. 그 외 날짜는 '수동 입력' 방식을 사용해 주세요.")
    else:
        st.success(f"현재 날씨 연동 ({source}): 기온 {temp}°C / 강수량 {rain}mm / 적설 {snow}mm")

    if st.button("날씨 새로고침 🔄"):
        st.session_state.pop("kma_weather", None)
        st.rerun()
else:
    temp = st.slider("기온 (°C)", min_value=-15.0, max_value=40.0, value=15.0, step=0.5)
    rain = st.slider("강수량 (mm)", min_value=0.0, max_value=100.0, value=0.0, step=0.5)
    snow = st.slider("적설 (mm)",  min_value=0.0, max_value=50.0,  value=0.0, step=0.5)

dt = datetime.combine(selected_date, datetime.min.time())
hol_name, hol_msg = check_holiday(dt)
if hol_msg:
    icon = "🎉" if hol_name else "ℹ️"
    st.markdown(f'<div class="holiday-banner">{icon} {hol_msg}</div>', unsafe_allow_html=True)

# ── 8. 6대 대시보드 탭 레이아웃 및 렌더러 호출 ─────────────────────────────────────
tab0, tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🚇 실시간 관제 맵", 
    "📊 시간대별 예측", 
    "🌦 날씨 시뮬레이터", 
    "🏆 역별 랭킹", 
    "⏰ 최적 탑승 시간",
    "🤖 모델별 예측치 비교"
])

# 탭 0: 실시간 관제 지도 시각화 및 지하철 위치 동기화
from components.map_tab import render_map_tab
with tab0:
    seoul_key = os.environ.get("SEOUL_SUBWAY_API_KEY", "sample")
    render_map_tab(selected_station, seoul_key)

# 탭 1 ~ 3: 예측 조회, 날씨 조건별 시뮬레이션, 역별 혼잡도 랭킹 렌더링
from components.forecast_tabs import render_forecast_tabs
render_forecast_tabs(
    tab1, tab2, tab3, selected_station, dt, selected_hour, 
    temp, rain, snow, active_model, ALL_MODELS, le_station, LSTM_BASE_DF, STATIONS
)

# 탭 4 ~ 5: 최적 시간 플래너 및 AI 모델별 다각적 예측 비교 렌더링
from components.comparison_tabs import render_comparison_tabs
render_comparison_tabs(
    tab4, tab5, selected_station, dt, selected_hour,
    temp, rain, snow, active_model, ALL_MODELS, le_station, LSTM_BASE_DF, STATIONS
)

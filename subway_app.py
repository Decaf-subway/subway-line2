"""
서울 지하철 2호선 실시간 관제 및 승하차 예측 앱
===============================================
실행: streamlit run subway_app.py
필요 패키지: streamlit xgboost scikit-learn joblib holidays pandas numpy plotly requests
"""

import os
# TensorFlow의 macOS Metal 플러그인 초기화 지연 방지 (CPU 모드 강제)
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import joblib
import holidays as hd
import requests
import json
import xml.etree.ElementTree as ET
from datetime import date, datetime, timedelta

# ---- PAGE CONFIGURATION --------------------------------------------------------
st.set_page_config(
    page_title="2호선 실시간 관제 및 승하차 예측",
    page_icon="🚇",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---- STYLING & CUSTOM CSS ------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@400;500;700&display=swap');
    html, body, [class*="css"] { font-family: 'Noto Sans KR', sans-serif; }
    .metric-card {
        background: #f8f9fa; border-radius: 12px; padding: 16px 20px;
        border: 1px solid #e9ecef; text-align: center;
    }
    .metric-label { font-size: 12px; color: #6c757d; margin-bottom: 4px; }
    .metric-value { font-size: 28px; font-weight: 700; color: #212529; }
    .metric-sub   { font-size: 12px; color: #6c757d; margin-top: 2px; }
    .badge-쾌적   { background:#d1fae5; color:#065f46; padding:4px 12px; border-radius:999px; font-size:13px; font-weight:500; }
    .badge-보통   { background:#dbeafe; color:#1e40af; padding:4px 12px; border-radius:999px; font-size:13px; font-weight:500; }
    .badge-혼잡   { background:#fef3c7; color:#92400e; padding:4px 12px; border-radius:999px; font-size:13px; font-weight:500; }
    .badge-매우혼잡 { background:#fee2e2; color:#991b1b; padding:4px 12px; border-radius:999px; font-size:13px; font-weight:500; }
    .holiday-banner {
        background: linear-gradient(135deg, #667eea22, #764ba222);
        border: 1px solid #667eea44; border-radius: 10px;
        padding: 12px 16px; margin-bottom: 16px;
        font-size: 14px; color: #4c3499;
    }
    .station-btn { cursor: pointer; transition: all 0.15s; }
    .rank-card {
        background: #fff; border: 1px solid #e9ecef; border-radius: 8px;
        padding: 10px 14px; margin-bottom: 6px;
        display: flex; justify-content: space-between; align-items: center;
    }
</style>
""", unsafe_allow_html=True)

# ---- LOAD ENVIRONMENT VARIABLES (.env) -----------------------------------------
if os.path.exists(".env"):
    with open(".env", "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ[k.strip()] = v.strip()

# ---- DEFAULT KMA WEATHER API KEY ----------------------------------------------
DEFAULT_KMA_KEY = "c8a63dac44b7ee08af43d9426d891501328ed2e638e432fe873ff9bb28d11484"

# ---- LOAD STATIC TRANSFER DATA (FROM JSON) ------------------------------------
TRANSFER_INFO = {}
if os.path.exists("data/transfer_info.json"):
    try:
        with open("data/transfer_info.json", "r", encoding="utf-8") as f:
            TRANSFER_INFO = json.load(f)
    except Exception as e:
        st.error(f"환승 정보 로드 오류: {e}")

# ---- LINE 2 STATIONS LIST (MAPPED IN CIRCULAR ORDER) ---------------------------
MAIN_LINE = [
    "시청", "을지로입구", "을지로3가", "을지로4가", "동대문역사문화공원(DDP)",
    "신당", "상왕십리", "왕십리", "한양대", "뚝섬", "성수", "건대입구",
    "구의", "강변", "잠실나루", "잠실", "잠실새내", "종합운동장", "삼성",
    "선릉", "역삼", "강남", "교대", "서초", "방배", "사당", "낙성대",
    "서울대입구", "봉천", "신림", "신대방", "구로디지털단지", "대림",
    "신도림", "문래", "영등포구청", "당산", "합정", "홍대입구", "신촌",
    "이대", "아현", "충정로"
]

SUNGSU_BRANCH = ["용답", "신답", "용두(동대문구청)", "신설동"]
SINDORIM_BRANCH = ["도림천", "양천구청", "신정네거리", "까치산"]

# Integrated list of all Line 2 main and branch stations
ALL_LINE2_STATIONS = MAIN_LINE + SUNGSU_BRANCH + SINDORIM_BRANCH

# ---- LOAD HISTORICAL TIME-SERIES DATA FOR LSTM -------------------------------
@st.cache_data
def load_lstm_base_dataset():
    path = "final_dataset_230101-241231.csv"
    if not os.path.exists(path):
        path = "final_dataset.csv"
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path)
        # Keep only required columns to optimize performance and memory
        df['날짜'] = pd.to_datetime(df['날짜'])
        df['요일'] = df['날짜'].dt.weekday
        df['요일_시간'] = df['요일'] * 24 + df['시간']
        
        # Sort by Date and Hour to maintain chronological time-series structure
        df = df.sort_values(by=['역명', '날짜', '시간']).reset_index(drop=True)
        
        # Group by station to calculate average boarding and alighting numbers for backup
        if '역별_평균_승차' not in df.columns:
            stn_in = df.groupby('역명')['승차인원'].mean().to_dict()
            df['역별_평균_승차'] = df['역명'].map(stn_in)
        if '역별_평균_하차' not in df.columns:
            stn_out = df.groupby('역명')['하차인원'].mean().to_dict()
            df['역별_평균_하차'] = df['역명'].map(stn_out)
            
        # Apply log transform (log1p)
        for col in ['승차인원', '하차인원', '역별_평균_승차', '역별_평균_하차']:
            df[col] = np.log1p(df[col])
            
        return df[['역명', '시간', '승차인원', '하차인원', '요일', '월', '공휴일여부', '기온', '강수량', '적설', '요일_시간', '역별_평균_승차', '역별_평균_하차']]
    except Exception:
        return None

LSTM_BASE_DF = load_lstm_base_dataset()

# ---- MODEL LOADING SECTION -----------------------------------------------------
import keras

@keras.saving.register_keras_serializable(package="Custom")
class PatchedEmbedding(keras.layers.Embedding):
    def __init__(self, *args, **kwargs):
        kwargs.pop('quantization_config', None)
        super().__init__(*args, **kwargs)

@st.cache_resource
def load_all_models():
    """
    Attempt to load all ML/DL models (XGBoost, LightGBM, RandomForest, LSTM).
    If a model file is missing, it will fall back to a deterministic dummy simulator.
    """
    models = {}
    le_station = None
    
    # 0. Load label encoder (for XGBoost and common UI categories)
    try:
        le_station = joblib.load("models/xgboost/label_encoder_station.pkl")
    except Exception as e:
        print(f"[WARN] Failed to load common label_encoder_station.pkl (XGBoost/LightGBM disabled): {e}")

    # 1. XGBoost
    try:
        models["XGBoost"] = {
            "board": joblib.load("models/xgboost/xgb_board_model.pkl"),
            "alight": joblib.load("models/xgboost/xgb_alight_model.pkl"),
            "loaded": True
        }
    except Exception as e:
        print(f"[WARN] Failed to load XGBoost model files: {e}")
        models["XGBoost"] = {"loaded": False}

    # 2. LightGBM
    try:
        models["LightGBM"] = {
            "board": joblib.load("models/lightgbm/lgb_boadin_model.pkl"),
            "alight": joblib.load("models/lightgbm/lgb_alight_model.pkl"),
            "loaded": True
        }
    except Exception as e:
        print(f"[WARN] Failed to load LightGBM model files: {e}")
        models["LightGBM"] = {"loaded": False}

    # 3. RandomForest
    try:
        models["RandomForest"] = {
            "board": joblib.load("models/randomforest/randomforest_boarding_model.pkl"),
            "alight": joblib.load("models/randomforest/randomforest_dropoff_model.pkl"),
            "cols": joblib.load("models/randomforest/model_boardin_columns.pkl"),
            "loaded": True
        }
    except Exception as e:
        print(f"[WARN] Failed to load RandomForest model files: {e}")
        models["RandomForest"] = {"loaded": False}

    # 4. LSTM
    try:
        import keras
        m_board = keras.models.load_model(
            "models/lstm/lstm_boarding.keras", 
            custom_objects={'Embedding': PatchedEmbedding}, 
            compile=False
        )
        m_alight = keras.models.load_model(
            "models/lstm/lstm_alighting.keras", 
            custom_objects={'Embedding': PatchedEmbedding}, 
            compile=False
        )
        
        # Try loading custom LSTM models for Jamsil station (fallback to general if failed)
        try:
            m_board_jamsil = keras.models.load_model(
                "models/lstm/lstm_boarding_잠실.keras", 
                custom_objects={'Embedding': PatchedEmbedding}, 
                compile=False
            )
            m_alight_jamsil = keras.models.load_model(
                "models/lstm/lstm_alighting_잠실.keras", 
                custom_objects={'Embedding': PatchedEmbedding}, 
                compile=False
            )
        except Exception:
            m_board_jamsil = m_board
            m_alight_jamsil = m_alight

        scaler_path = "models/lstm/scaler.pkl"
        if not os.path.exists(scaler_path):
            scaler_path = "models/lstm/scaler_x.pkl"
            
        scaler = joblib.load(scaler_path)
        le_lstm = joblib.load("models/lstm/label_encoder.pkl") # Load LSTM specific label encoder

        models["LSTM"] = {
            "board": m_board,
            "alight": m_alight,
            "board_jamsil": m_board_jamsil,
            "alight_jamsil": m_alight_jamsil,
            "scaler": scaler,
            "le": le_lstm,
            "loaded": True
        }
    except Exception as e:
        models["LSTM"] = {"loaded": False}

    return models, le_station, le_station is not None

ALL_MODELS, le_station, model_loaded = load_all_models()
model_loaded_state = any(m.get("loaded", False) for m in ALL_MODELS.values())

# Load all station names supported by the model (including branch line stations like Sindap, Yongdap)
if model_loaded:
    STATIONS = sorted(list(le_station.classes_))
else:
    STATIONS = sorted(ALL_LINE2_STATIONS)

# Set average boarding passengers per station (for congestion baseline calculation)
STATION_AVG = {stn: 1500 for stn in STATIONS}
STATION_AVG.update({
    "강남": 3200, "홍대입구": 2800, "잠실": 2600, "신림": 2200,
    "신도림": 2400, "건대입구": 2000, "사당": 2100, "왕십리": 1800,
    "선릉": 1900, "역삼": 1800, "교대": 1600, "합정": 1500,
    "시청": 1200, "을지로입구": 1400, "신설동": 1100, "용두(동대문구청)": 600,
    "신답": 400, "용답": 700, "까치산": 1800, "신정네거리": 1200,
    "양천구청": 900, "도림천": 300
})

# ---- FEATURE COLUMNS DEFINITION ------------------------------------------------
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

# ---- MODEL PREDICTION FUNCTIONS ------------------------------------------------
def get_holiday_type(d, kr_hols):
    if d not in kr_hols:
        return 0
    name = kr_hols.get(d, "")
    # Lunar New Year / Chuseok holidays (preceding/following days included)
    if any(k in name for k in ["Korean New Year", "Chuseok", "설날", "추석", "preceding", "second day"]):
        return 3
    # Family and outdoor activity holidays
    elif any(k in name for k in ["Children", "Christmas", "Buddha", "어린이날", "성탄절", "기독탄신일", "부처님오신날", "석가탄신일"]):
        return 2
    return 1

def predict(station, dt, hour, temp, rain=0.0, snow=0.0, model_name="XGBoost"):
    # Check if the selected model's data is successfully loaded
    m_info = ALL_MODELS.get(model_name, {})
    if not m_info.get("loaded", False):
        raise ValueError(f"Model '{model_name}' is not loaded. Please verify model files in models/ directory.")

    # Real model prediction logic if files exist
    kr_hols  = hd.KR(years=dt.year)
    is_hol   = int(dt.date() in kr_hols)
    stn_enc  = le_station.transform([station])[0]
    prev_d   = dt - timedelta(days=1)
    next_d   = dt + timedelta(days=1)
    prev_off = (prev_d.weekday() >= 5) or (prev_d.date() in kr_hols)
    next_off = (next_d.weekday() >= 5) or (next_d.date() in kr_hols)
    is_off   = (dt.weekday() >= 5) or bool(is_hol)

    row = {
        "역명_enc":      stn_enc,
        "시간":          hour,
        "시간_sin":      np.sin(2 * np.pi * hour / 24),
        "시간_cos":      np.cos(2 * np.pi * hour / 24),
        "요일":          dt.weekday(),
        "요일_sin":      np.sin(2 * np.pi * dt.weekday() / 7),
        "요일_cos":      np.cos(2 * np.pi * dt.weekday() / 7),
        "월":            dt.month,
        "월_sin":        np.sin(2 * np.pi * dt.month / 12),
        "월_cos":        np.cos(2 * np.pi * dt.month / 12),
        "공휴일여부":     is_hol,
        "비근무일":       int(dt.weekday() >= 5 or bool(is_hol)),
        "출근피크":       int(hour in [7, 8, 9]),
        "퇴근피크":       int(hour in [18, 19, 20]),
        "기온":           temp,
        "강수량":         rain,
        "적설":           snow,
        "강수_여부":      int(rain > 0),
        "적설_여부":      int(snow > 0),
        "불쾌지수":       round(9/5*temp - 0.55*(1-int(rain>0)*0.8)*(9/5*temp-26)+32, 2),
        "공휴일_유형":    get_holiday_type(dt.date(), kr_hols),
        "연휴_여부":      int(is_off and (prev_off or next_off)),
        "공휴일_전날":    int(next_d.date() in kr_hols),
        "공휴일_다음날":  int(prev_d.date() in kr_hols),
    }

    X = pd.DataFrame([row])[FEATURE_COLS]
    
    # LSTM Model specific inference execution branch (requires Keras sequence input)
    if model_name == "LSTM" and m_info.get("loaded", False):
        try:
            scaler = m_info["scaler"]
            
            # Switch models for Jamsil station (special customized version)
            is_jamsil = "잠실" in station
            if is_jamsil and "board_jamsil" in m_info and "alight_jamsil" in m_info:
                m_board = m_info["board_jamsil"]
                m_alight = m_info["alight_jamsil"]
            else:
                m_board = m_info["board"]
                m_alight = m_info["alight"]

            # Extract station label encoding using LSTM specific label encoder (fuzzy matching)
            station_idx = 0
            le_lstm = m_info.get("le")
            if le_lstm is not None:
                for idx, cls in enumerate(le_lstm.classes_):
                    if station in cls or cls in station:
                        station_idx = idx
                        break

            # Feature schema for MinMaxScaler: ['승차인원', '하차인원', '시간', '요일', '월', '공휴일여부', '기온', '강수량', '적설', '요일_시간', '역별_평균_승차', '역별_평균_하차']
            avg_val = 1500
            if le_lstm is not None:
                matched_cls = le_lstm.classes_[station_idx]
                for k, v in STATION_AVG.items():
                    if k in matched_cls or matched_cls in k:
                        avg_val = v
                        break
                
            # The new scaler expects log1p transformed values for passengers and averages.
            log_avg_val = np.log1p(avg_val)

            # Construct temporary DataFrame for scaling (using log-transformed passenger baseline)
            row_scaler = {
                "승차인원": 0.0,
                "하차인원": 0.0,
                "시간": float(hour),
                "요일": float(dt.weekday()),
                "월": float(dt.month),
                "공휴일여부": float(dt.date() in hd.KR(years=dt.year)),
                "기온": float(temp),
                "강수량": float(rain),
                "적설": float(snow),
                "요일_시간": float(dt.weekday() * 24 + hour),
                "역별_평균_승차": float(log_avg_val),
                "역별_평균_하차": float(log_avg_val)
            }
            
            # Align column order before applying transform
            scaler_cols = ['승차인원', '하차인원', '시간', '요일', '월', '공휴일여부', '기온', '강수량', '적설', '요일_시간', '역별_평균_승차', '역별_평균_하차']
            df_scale_in = pd.DataFrame([row_scaler])[scaler_cols]
            df_scaled = pd.DataFrame(scaler.transform(df_scale_in), columns=scaler_cols)
            
            # Build 12 feature list complying with feature_info.json specification
            feat_12 = list(df_scaled.values[0])
            
            # Construct time-series sequence data from historical dataset
            if LSTM_BASE_DF is not None:
                matched_stn = station
                le_lstm = m_info.get("le")
                if le_lstm is not None:
                    matched_stn = le_lstm.classes_[station_idx]
                
                station_all = LSTM_BASE_DF[LSTM_BASE_DF['역명'] == matched_stn].copy().reset_index(drop=True)
                
                same_hour_idx = station_all[
                    station_all['시간'] == hour
                ].index.tolist()
                
                if len(same_hour_idx) == 0:
                    same_hour_idx = [np.abs(station_all['시간'] - hour).idxmin()]
                    
                best_idx = same_hour_idx[-1]
                start_idx = max(0, best_idx - 12)
                seq_data = station_all.iloc[start_idx:best_idx]
                
                if len(seq_data) < 12:
                    seq_data = station_all.iloc[best_idx:best_idx + 12]
                    
                seq_raw = seq_data.tail(12)[scaler_cols].values.copy()
                seq = scaler.transform(seq_raw)
                seq[-1] = feat_12
            else:
                seq = np.array([feat_12] * 12)
                
            X_in = np.expand_dims(seq, axis=0)
            
            # Construct station ID input array -> shape: (1, 1)
            stn_in = np.array([[float(station_idx)]])
            
            # Run model inference with multi-inputs [sequence_input, station_input]
            pred_b = np.array(m_board([X_in, stn_in], training=False))
            pred_a = np.array(m_alight([X_in, stn_in], training=False))
            
            # Inverse scale the predicted outputs (reshape to 12D array)
            # Inverse scaling for boarding passengers
            dummy_row_b = np.zeros((1, 12))
            dummy_row_b[0, 0] = pred_b.flatten()[0] # 승차인원 위치
            log_val_b = scaler.inverse_transform(dummy_row_b)[0, 0]
            
            # Inverse scaling for alighting passengers
            dummy_row_a = np.zeros((1, 12))
            dummy_row_a[0, 1] = pred_a.flatten()[0] # 하차인원 위치
            log_val_a = scaler.inverse_transform(dummy_row_a)[0, 1]
            
            # Apply exponential inverse (expm1) to restore to original passenger count
            b_val = np.expm1(log_val_b)
            a_val = np.expm1(log_val_a)
            
            b = int(np.clip(b_val, 0, None))
            a = int(np.clip(a_val, 0, None))
            
        except Exception as e:
            # Fall back to XGBoost model predictions in case of LSTM runtime errors
            xgb_info = ALL_MODELS.get("XGBoost", {})
            if xgb_info.get("loaded", False):
                b = int(np.clip(xgb_info["board"].predict(X), 0, None)[0])
                a = int(np.clip(xgb_info["alight"].predict(X), 0, None)[0])
            else:
                raise RuntimeError(f"LSTM prediction failed: {e}, and XGBoost fallback is unavailable.")
    else:
        if model_name == "RandomForest":
            # Build 57-dimensional one-hot encoded feature DataFrame for RandomForest
            rf_cols = m_info["cols"]
            row_rf = {
                "시간": float(hour),
                "요일": float(dt.weekday()),
                "월": float(dt.month),
                "공휴일여부": float(is_hol),
                "기온": float(temp),
                "강수량": float(rain),
                "적설": float(snow)
            }
            # Set values for the station name one-hot columns
            for col in rf_cols:
                if col.startswith("역명_"):
                    stn_part = col.replace("역명_", "")
                    row_rf[col] = 1.0 if stn_part == station else 0.0
            
            X_rf = pd.DataFrame([row_rf])[rf_cols]
            b = int(np.clip(m_info["board"].predict(X_rf),  0, None)[0])
            a = int(np.clip(m_info["alight"].predict(X_rf), 0, None)[0])
        elif model_name == "LightGBM":
            # Build 12-dimensional feature DataFrame for LightGBM with categorical mappings
            row_lgb = {
                "역명": station,
                "호선": "2호선",
                "시간": int(hour),
                "요일": int(dt.weekday()),
                "월": int(dt.month),
                "공휴일여부": int(is_hol),
                "기온": float(temp),
                "강수량": float(rain),
                "적설": float(snow),
                "year": int(dt.year),
                "day": int(dt.day),
                "weekday": int(dt.weekday())
            }
            df_lgb = pd.DataFrame([row_lgb])
            
            # Align category mapping order for categorical variables
            stn_categories = sorted(list(le_station.classes_))
            line_categories = ['2호선']
            df_lgb["역명"] = pd.Categorical(df_lgb["역명"], categories=stn_categories)
            df_lgb["호선"] = pd.Categorical(df_lgb["호선"], categories=line_categories)
            
            lgb_cols = ['역명', '호선', '시간', '요일', '월', '공휴일여부', '기온', '강수량', '적설', 'year', 'day', 'weekday']
            X_lgb = df_lgb[lgb_cols]
            
            b = int(np.clip(m_info["board"].predict(X_lgb),  0, None)[0])
            a = int(np.clip(m_info["alight"].predict(X_lgb), 0, None)[0])
        else:
            # Standard XGBoost inference path
            b = int(np.clip(m_info["board"].predict(X),  0, None)[0])
            a = int(np.clip(m_info["alight"].predict(X), 0, None)[0])
        
    return b, a

def predict_day(station, dt, temp, rain=0.0, snow=0.0, model_name="XGBoost"):
    hours, boards, alights = [], [], []
    for hr in range(5, 24):
        b, a = predict(station, dt, hr, temp, rain, snow, model_name)
        hours.append(hr)
        boards.append(b)
        alights.append(a)
    return hours, boards, alights

def get_congestion(board, station):
    avg = STATION_AVG.get(station, 1500)
    ratio = board / avg if avg > 0 else 1
    if ratio < 0.6:   return "쾌적",   "#065f46", "#d1fae5"
    if ratio < 1.0:   return "보통",   "#1e40af", "#dbeafe"
    if ratio < 1.5:   return "혼잡",   "#92400e", "#fef3c7"
    return              "매우혼잡", "#991b1b", "#fee2e2"

# ---- WEATHER API CLIENTS -------------------------------------------------------
def get_kma_weather(api_key, dt, hour):
    """
    KMA short-term forecast API (Seoul base nx=60, ny=127).
    Used for retrieving weather forecast within 3 days.
    """
    try:
        base_times = [2, 5, 8, 11, 14, 17, 20, 23]
        now = datetime.now()
        base_hour = max([t for t in base_times if t <= now.hour], default=23)
        base_time = f"{base_hour:02d}00"
        base_date = now.strftime("%Y%m%d")

        url = (
            "http://apis.data.go.kr/1360000/VilageFcstInfoService_2.0/getVilageFcst"
            f"?serviceKey={api_key}"
            f"&pageNo=1&numOfRows=1000&dataType=JSON"
            f"&base_date={base_date}&base_time={base_time}"
            f"&nx=60&ny=127"
        )

        res = requests.get(url, timeout=10)
        if res.status_code != 200:
            return None

        data = res.json()
        if "response" not in data or "body" not in data["response"] or "items" not in data["response"]["body"]:
            return None

        items = data["response"]["body"]["items"]["item"]

        target_date = dt.strftime("%Y%m%d")
        target_time = f"{hour:02d}00"

        filtered = [
            x for x in items
            if x["fcstDate"] == target_date and x["fcstTime"] == target_time
        ]

        if not filtered:
            return None

        result = {}
        for item in filtered:
            result[item["category"]] = item["fcstValue"]

        # Helper function to parse KMA forecast string values
        def parse_kma_value(val_str, default=0.0):
            if not val_str:
                return default
            val_str = str(val_str).strip()
            for w in ["강수없음", "적설없음", "mm", "cm"]:
                val_str = val_str.replace(w, "0")
            val_str = val_str.replace("1mm 미만", "0.5").replace("1cm 미만", "0.5")
            try:
                return float(val_str)
            except ValueError:
                return default

        # Parse temperature, precipitation, and snowfall values
        temp = float(result.get("TMP", 15.0))
        rain = parse_kma_value(result.get("PCP", "0"))
        snow = parse_kma_value(result.get("SNO", "0"))

        return round(temp, 1), round(rain, 1), round(snow, 1), "단기예보"

    except Exception as e:
        print(f"[WARN] Exception in get_kma_weather: {e}")
        return None

def get_mid_forecast_time():
    """Get release time of KMA mid-term forecast (06:00, 18:00)"""
    now = datetime.now()
    if now.hour < 6:
        base_date = (now - timedelta(days=1)).strftime("%Y%m%d")
        tm_fc = base_date + "1800"
    elif now.hour < 18:
        base_date = now.strftime("%Y%m%d")
        tm_fc = base_date + "0600"
    else:
        base_date = now.strftime("%Y%m%d")
        tm_fc = base_date + "1800"
    return tm_fc

def get_kma_mid_weather(api_key, dt):
    """
    KMA mid-term forecast API (Seoul land-code 11B00000, temperature-point 11B10101).
    Used for retrieving weather forecast between 3 to 10 days.
    """
    try:
        # Secure target date as date object for calculation
        target_date = dt if isinstance(dt, date) else dt.date()
        days_diff = (target_date - date.today()).days
        if days_diff < 3 or days_diff > 10:
            return None

        tm_fc = get_mid_forecast_time()

        # 1. Mid-term temperature forecast lookup (Seoul code 11B10101)
        url_ta = (
            "http://apis.data.go.kr/1360000/MidFcstInfoService/getMidTa"
            f"?serviceKey={api_key}"
            f"&pageNo=1&numOfRows=10&dataType=JSON"
            f"&regId=11B10101"
            f"&tmFc={tm_fc}"
        )

        # 2. Mid-term land weather forecast lookup (Seoul/Gyeonggi code 11B00000)
        url_land = (
            "http://apis.data.go.kr/1360000/MidFcstInfoService/getMidLandFcst"
            f"?serviceKey={api_key}"
            f"&pageNo=1&numOfRows=10&dataType=JSON"
            f"&regId=11B00000"
            f"&tmFc={tm_fc}"
        )

        res_ta = requests.get(url_ta, timeout=10)
        res_land = requests.get(url_land, timeout=10)

        if res_ta.status_code != 200 or res_land.status_code != 200:
            return None

        data_ta = res_ta.json()
        data_land = res_land.json()

        if "response" not in data_ta or "body" not in data_ta["response"] or "items" not in data_ta["response"]["body"]:
            return None
        if "response" not in data_land or "body" not in data_land["response"] or "items" not in data_land["response"]["body"]:
            return None

        item_ta = data_ta["response"]["body"]["items"]["item"][0]
        item_land = data_land["response"]["body"]["items"]["item"][0]

        # Temperature
        ta_min = float(item_ta.get(f"taMin{days_diff}", 10.0))
        ta_max = float(item_ta.get(f"taMax{days_diff}", 20.0))
        avg_temp = (ta_min + ta_max) / 2.0

        # Weather conditions and precipitation probability
        if days_diff in [8, 9, 10]:
            wf = item_land.get(f"wf{days_diff}", "맑음")
            rn_st = float(item_land.get(f"rnSt{days_diff}", 0))
        else:
            wf_am = item_land.get(f"wf{days_diff}Am", "맑음")
            wf_pm = item_land.get(f"wf{days_diff}Pm", "맑음")
            wf = f"{wf_am} / {wf_pm}"
            rn_am = float(item_land.get(f"rnSt{days_diff}Am", 0))
            rn_pm = float(item_land.get(f"rnSt{days_diff}Pm", 0))
            rn_st = (rn_am + rn_pm) / 2.0

        rain = 0.0
        snow = 0.0
        if "비" in wf:
            rain = 5.0
        if "눈" in wf:
            snow = 2.0

        return round(avg_temp, 1), round(rain, 1), round(snow, 1), f"{wf} (강수확률 {int(rn_st)}%)"

    except Exception as e:
        return None

def get_weather_with_fallback(api_key, dt, hour):
    """
    Chained query for short-term and mid-term KMA weather forecast.
    Falls back to monthly local average temperature if API fails.
    Returns: (temp, rain, snow, source_name)
    """
    # 0. Calculate difference in days to determine forecast range
    target_date = dt if isinstance(dt, date) else dt.date()
    days_diff = (target_date - date.today()).days
    
    # 1. Short-term forecast (within 3 days)
    if 0 <= days_diff <= 2:
        res = get_kma_weather(api_key, dt, hour)
        if res is not None:
            return res
            
    # 2. Mid-term forecast (from 3 to 10 days)
    if 3 <= days_diff <= 10:
        res_mid = get_kma_mid_weather(api_key, dt)
        if res_mid is not None:
            return res_mid
            
    # 3. Fall back to historical monthly average temperatures if out of range or API fails
    month = dt.month
    monthly_temps = {
        1: -2.5, 2: 0.0, 3: 5.5, 4: 12.0, 5: 18.0, 6: 22.5,
        7: 25.5, 8: 26.5, 9: 21.0, 10: 14.5, 11: 7.0, 12: 0.0
    }
    temp = monthly_temps.get(month, 15.0)
    source_name = f"기본값 (로컬 {month}월 평균 기온 대체)"
    if days_diff < 0 or days_diff > 10:
        source_name = f"기본값 (날짜 범위 초과 대체)"
    
    return temp, 0.0, 0.0, source_name


# ---- SEOUL METRO REAL-TIME TRAIN POSITION API ----------------------------------────────────────
def get_realtime_train_positions(api_key="sample"):
    """Seoul Open Data Plaza Line 2 real-time train positions API"""
    try:
        url = f"http://swopenAPI.seoul.go.kr/api/subway/{api_key}/json/realtimePosition/0/50/2호선"
        res = requests.get(url, timeout=5)
        if res.status_code == 200:
            data = res.json()
            if "realtimePositionList" in data:
                return data["realtimePositionList"]
    except Exception as e:
        pass
    return []

def get_realtime_station_arrival(station_name, api_key="sample"):
    """Seoul Open Data Plaza real-time arrival info API for a specific station"""
    try:
        # Strip brackets from station name to query raw arrival info
        pure_name = station_name.split("(")[0]
        url = f"http://swopenAPI.seoul.go.kr/api/subway/{api_key}/json/realtimeStationArrival/0/20/{pure_name}"
        res = requests.get(url, timeout=5)
        if res.status_code == 200:
            data = res.json()
            if "realtimeArrivalList" in data:
                # Filter for Line 2 arrivals (subwayId: 1002)
                return [arr for arr in data["realtimeArrivalList"] if arr.get("subwayId") == "1002"]
    except Exception as e:
        pass
    return []

# ---- HOLIDAY DETECTION ---------------------------------------------------------
def check_holiday(dt):
    kr_hols = hd.KR(years=dt.year)
    if dt.date() in kr_hols:
        name = kr_hols.get(dt.date(), "공휴일")
        htype = get_holiday_type(dt.date(), kr_hols)
        if htype == 3:
            return name, "설날·추석 연휴입니다. 평소보다 30~50% 한산할 것으로 예상됩니다."
        return name, "공휴일입니다. 평소보다 20~40% 한산할 것으로 예상됩니다."
    prev_d = dt - timedelta(days=1)
    next_d = dt + timedelta(days=1)
    if next_d.date() in kr_hols:
        return None, f"내일은 {kr_hols.get(next_d.date())}입니다. 귀성 인파로 일부 역이 혼잡할 수 있습니다."
    if prev_d.date() in kr_hols:
        return None, f"어제가 공휴일이었습니다. 귀경 인파가 있을 수 있습니다."
    return None, None

# ═══════════════════════════════════════════════════════════════════════════════
# UI 및 사이드바 설정
# ═══════════════════════════════════════════════════════════════════════════════

st.title("🚇 서울 지하철 2호선 실시간 관제 및 승하차 예측")
if not model_loaded_state:
    st.error("❌ 필수 AI 모델 파일을 전혀 찾을 수 없습니다. `models/` 디렉토리에 학습 완료된 모델 파일을 안전하게 배치해 주세요.")
    st.stop()

# ---- SIDEBAR SETTINGS ----------------------------------------------------------
with st.sidebar:
    st.header("⚙️ 예측 설정")

    # Filter and display only successfully loaded models in sidebar
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

# 기상청 날씨 로드
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

# ---- HOLIDAY BANNER ------------------------------------------------------------
dt = datetime.combine(selected_date, datetime.min.time())
hol_name, hol_msg = check_holiday(dt)
if hol_msg:
    icon = "🎉" if hol_name else "ℹ️"
    st.markdown(f'<div class="holiday-banner">{icon} {hol_msg}</div>', unsafe_allow_html=True)

# ---- TAB NAVIGATION CONFIGURATION ----------------------------------------------
tab0, tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🚇 실시간 관제 맵", 
    "📊 시간대별 예측", 
    "🌦 날씨 시뮬레이터", 
    "🏆 역별 랭킹", 
    "⏰ 최적 탑승 시간",
    "🤖 모델별 예측치 비교"
])

# ================================================================================
# TAB 0: Real-time Control Map (Circular metro line visualizer)
# ================================================================================
with tab0:
    st.subheader("🚇 2호선 실시간 열차 위치 관제 모니터")
    
    col_ref, col_lbl = st.columns([4, 1])
    with col_ref:
        st.caption("서울 열린데이터광장 실시간 열차 위치 API 연동")
    with col_lbl:
        if st.button("실시간 위치 새로고침 🔄"):
            st.rerun()

    # Build coordinate mapping dictionary for 51 circular line stations
    N_main = len(MAIN_LINE)
    R_main = 18.0
    station_coords = {}

    for i, name in enumerate(MAIN_LINE):
        # 12시 방향부터 시계방향 회전
        theta = np.pi / 2 - (2 * np.pi * i / N_main)
        x = R_main * np.cos(theta)
        y = R_main * np.sin(theta)
        station_coords[name] = (x, y, "main")

    # Sungsu branch coordinates (stretching vertically north from Sungsu)
    x_sungsu, y_sungsu = station_coords["성수"][0], station_coords["성수"][1]
    for i, name in enumerate(SUNGSU_BRANCH):
        station_coords[name] = (x_sungsu, y_sungsu + (i + 1) * 2.5, "sungsu")

    # Sindorim branch coordinates (stretching southwest from Sindorim)
    x_sindorim, y_sindorim = station_coords["신도림"][0], station_coords["신도림"][1]
    for i, name in enumerate(SINDORIM_BRANCH):
        station_coords[name] = (x_sindorim - (i + 1) * 2.0, y_sindorim - (i + 1) * 2.0, "sindorim")

    # Collect real-time train location data
    seoul_key = os.environ.get("SEOUL_SUBWAY_API_KEY", "sample")
    trains = get_realtime_train_positions(seoul_key)

    # Initialize Plotly figure
    fig_map = go.Figure()

    # 1. Draw main Loop Line 2 circular track with outer glow
    main_x = [station_coords[name][0] for name in MAIN_LINE] + [station_coords[MAIN_LINE[0]][0]]
    main_y = [station_coords[name][1] for name in MAIN_LINE] + [station_coords[MAIN_LINE[0]][1]]
    
    # 본선 아우터 글로우 (연한 녹색 번짐 효과)
    fig_map.add_trace(go.Scatter(
        x=main_x, y=main_y,
        mode="lines",
        line=dict(color="rgba(34, 197, 94, 0.15)", width=9),
        hoverinfo="skip",
        showlegend=False
    ))
    # 본선 이너 메인선 (선명한 2호선 초록색)
    fig_map.add_trace(go.Scatter(
        x=main_x, y=main_y,
        mode="lines",
        line=dict(color="#22c55e", width=4),
        hoverinfo="skip",
        showlegend=False
    ))

    # 2. Draw Sungsu branch track
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

    # 3. Draw Sindorim branch track
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

    # 4. Add station nodes and typographic text layout (expanded font sizes for readability)
    node_x = [coord[0] for coord in station_coords.values()]
    node_y = [coord[1] for coord in station_coords.values()]
    node_names = list(station_coords.keys())
    
    node_symbols = []
    node_colors = []
    node_sizes = []
    node_border_colors = []
    node_border_widths = []
    
    node_text = []
    node_font_sizes = []
    
    # Define major hub stations for typographic hierarchy
    hubs = ["강남", "잠실", "홍대입구", "신도림", "사당", "신림", "시청", "건대입구", "성수", "왕십리", "선릉", "역삼", "교대"]
    
    for name in node_names:
        is_selected = (name == selected_station) or (selected_station.split("(")[0] in name and name.split("(")[0] in selected_station)
        is_hub = any(h in name for h in hubs)
        transfers = TRANSFER_INFO.get(name, [])
        
        clean_name = name.split("(")[0]
        
        # 1) Apply node styling
        if is_selected:
            node_symbols.append("circle")
            node_colors.append("#0ea5e9") # 스카이 블루 (선택역)
            node_sizes.append(15)
            node_border_colors.append("#ffffff")
            node_border_widths.append(2.5)
            
            # 선택역은 진하게 파란색 강조 텍스트 (글자 크기 14.0)
            node_text.append(f"<b><span style='color:#0ea5e9;'>{clean_name}</span></b>")
            node_font_sizes.append(14.0)
        elif transfers and any(transfers):
            node_symbols.append("square")
            node_colors.append("#ffffff") # 환승 허브역
            node_sizes.append(10)
            node_border_colors.append("#22c55e")
            node_border_widths.append(1.5)
            
            if is_hub:
                node_text.append(f"<b>{clean_name}</b>")
                node_font_sizes.append(11.5)
            else:
                node_text.append(clean_name)
                node_font_sizes.append(10.5)
        else:
            node_symbols.append("circle")
            node_colors.append("#22c55e") # 일반역
            node_sizes.append(7.0)
            node_border_colors.append("#ffffff")
            node_border_widths.append(1.0)
            
            if is_hub:
                node_text.append(f"<b>{clean_name}</b>")
                node_font_sizes.append(11.5)
            else:
                node_text.append(clean_name)
                node_font_sizes.append(10.5)

    # 노드 플롯 추가 (텍스트 색상은 정의하지 않고 테마 자동 상속, customdata 주입으로 클릭 이벤트 바인딩)
    fig_map.add_trace(go.Scatter(
        x=node_x, y=node_y,
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
        hoverinfo="text",
        hovertext=[f"역명: {name}" + (f"<br>환승: {', '.join(TRANSFER_INFO[name])}" if TRANSFER_INFO.get(name) and any(TRANSFER_INFO[name]) else "") for name in node_names],
        customdata=node_names,  # 클릭 연동을 위해 customdata 속성에 역명 주입
        showlegend=False
    ))

    # 5. Add real-time train markers
    train_x = []
    train_y = []
    train_hover = []
    train_colors = []
    train_symbols = []

    stt_map = {"0": "진입 중", "1": "도착 (정차 중)", "2": "출발 (주행 중)"}

    for t in trains:
        station_name = t.get("statnNm", "")
        target_stn = next((s for s in STATIONS if station_name in s), None)
        
        if target_stn and target_stn in station_coords:
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
                f"방향: <span style='color:#3b82f6;'>{direction}</span><br>"
                f"현재 위치: <b>{target_stn}역</b> ({status})<br>"
                f"행선지: <span style='color:#ef4444;'>{t.get('statnTnm', '순환')}행</span>"
            )
            # 네온 컬러 스키마 지정 (상행: 밝은 파랑, 하행: 오렌지)
            train_colors.append("#3b82f6" if is_up else "#f97316")
            train_symbols.append("triangle-up" if is_up else "triangle-down")

    if train_x:
        fig_map.add_trace(go.Scatter(
            x=train_x, y=train_y,
            mode="markers",
            marker=dict(
                symbol=train_symbols, 
                size=13, 
                color=train_colors, 
                line=dict(color="#ffffff", width=1.5)
            ),
            hoverinfo="text",
            hovertext=train_hover,
            name="실시간 열차"
        ))

    # Upgrade layout parameters (transparent backgrounds to inherit Streamlit themes)
    fig_map.update_layout(
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-28, 28]),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-28, 28]),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=5, r=5, t=5, b=5),
        height=800,
        legend=dict(
            orientation="h", 
            yanchor="bottom", 
            y=0.02, 
            xanchor="center", 
            x=0.5,
            font=dict(size=11),
            bgcolor="rgba(255, 255, 255, 0.7)",
            bordercolor="rgba(0,0,0,0.1)",
            borderwidth=1
        ),
    )
    
    # Activate on_select="rerun" in st.plotly_chart to capture click events
    selection = st.plotly_chart(fig_map, use_container_width=True, on_select="rerun")
    
    # Map marker click event handler
    if selection and "selection" in selection and "points" in selection["selection"]:
        points = selection["selection"]["points"]
        if points:
            clicked_point = points[0]
            # Check if customdata (station name) exists in the clicked marker
            if "customdata" in clicked_point and clicked_point["customdata"]:
                clicked_station = clicked_point["customdata"]
                # 세션 데이터와 상이할 경우 업데이트 후 즉시 화면을 갱신(Rerun)
                if clicked_station != st.session_state.get("selected_station"):
                    st.session_state["selected_station"] = clicked_station
                    st.rerun()

    st.markdown("⚠️ 실시간 열차 마커의 삼각형 방향은 **상행/내선순환(▲, 파랑)** 및 **하행/외선순환(▼, 주황)**을 나타내며, 외곽선이 굵은 하늘색 노드는 **현재 설정된 역**을 표기합니다. <b>지도의 역 마커를 직접 클릭하여 관심 역을 즉시 조회할 수 있습니다.</b>", unsafe_allow_html=True)

    # 6. Render real-time arrival list for selected station
    st.markdown("---")
    st.subheader(f"⏱️ {selected_station}역 실시간 도착 정보")
    
    with st.spinner(f"{selected_station}역 실시간 도착 정보 조회 중..."):
        arrival_data = get_realtime_station_arrival(selected_station, seoul_key)
        
    if arrival_data:
        arr_cols = st.columns(2)
        
        # 내선순환(상행) 및 외선순환(하행) 그룹핑
        up_trains = [a for a in arrival_data if a.get("updnLine") in ["내선", "상행"]]
        down_trains = [a for a in arrival_data if a.get("updnLine") in ["외선", "하행"]]
        
        # 내선순환 표시
        with arr_cols[0]:
            st.markdown("### 🔵 내선순환 (상행)")
            if up_trains:
                for idx, arr in enumerate(up_trains):
                    # 도착 메시지 (ex: "3분 20초 후", "전역 도착")
                    msg = arr.get("arvlMsg2", "정보 없음")
                    train_no = arr.get("btrainNo", "미정")
                    dest = arr.get("btrainSttus", "순환")  # 행선지
                    dest_nm = arr.get("trainLineNm", "내선순환")
                    
                    st.markdown(
                        f"""
                        <div style="background:#f1f5f9; padding:10px 15px; border-radius:8px; margin-bottom:8px; border-left:4px solid #3b82f6;">
                            <span style="font-weight:700; color:#1e293b;">{dest_nm}</span> (열차 {train_no}호)<br>
                            <span style="font-size:18px; font-weight:700; color:#3b82f6;">{msg}</span>
                        </div>
                        """, 
                        unsafe_allow_html=True
                    )
            else:
                st.info("현재 조회 가능한 내선순환 열차가 없습니다.")
                
        # 외선순환 표시
        with arr_cols[1]:
            st.markdown("### 🟠 외선순환 (하행)")
            if down_trains:
                for idx, arr in enumerate(down_trains):
                    msg = arr.get("arvlMsg2", "정보 없음")
                    train_no = arr.get("btrainNo", "미정")
                    dest = arr.get("btrainSttus", "순환")
                    dest_nm = arr.get("trainLineNm", "외선순환")
                    
                    st.markdown(
                        f"""
                        <div style="background:#f1f5f9; padding:10px 15px; border-radius:8px; margin-bottom:8px; border-left:4px solid #ef9f27;">
                            <span style="font-weight:700; color:#1e293b;">{dest_nm}</span> (열차 {train_no}호)<br>
                            <span style="font-size:18px; font-weight:700; color:#ef9f27;">{msg}</span>
                        </div>
                        """, 
                        unsafe_allow_html=True
                    )
            else:
                st.info("현재 조회 가능한 외선순환 열차가 없습니다.")
    else:
        st.info("실시간 도착 정보가 존재하지 않거나 가져오는데 실패했습니다.")

# ════════════════════════════════════════════════════════════════════════════════
# TAB 1: 시간대별 예측
# ════════════════════════════════════════════════════════════════════════════════
with tab1:
    col_info, col_metric = st.columns([3, 2])

    with col_info:
        st.subheader(f"🚉 {selected_station}역 · {selected_date.strftime('%Y년 %m월 %d일')}")
        day_names = ["월", "화", "수", "목", "금", "토", "일"]
        day_str = day_names[dt.weekday()]
        st.caption(f"{day_str}요일 · 기온 {temp}°C · 강수량 {rain}mm · 적설 {snow}mm")

    # 선택 시간 예측
    board_now, alight_now = predict(selected_station, dt, selected_hour, temp, rain, snow, active_model)
    cong_label, cong_color, cong_bg = get_congestion(board_now, selected_station)

    with col_metric:
        st.markdown(f"""
        <div style="background:{cong_bg}; border-radius:10px; padding:12px 16px; text-align:center; border:1px solid {cong_color}33;">
            <div style="font-size:12px; color:{cong_color}; margin-bottom:4px;">{selected_hour}시 혼잡도 ({active_model})</div>
            <div style="font-size:22px; font-weight:700; color:{cong_color};">{cong_label}</div>
        </div>
        """, unsafe_allow_html=True)

    # 선택 시간 메트릭
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

    # 하루 전체 예측 그래프
    hours, boards, alights = predict_day(selected_station, dt, temp, rain, snow, active_model)

    fig = go.Figure()

    # 출근 피크 음영
    fig.add_vrect(
        x0=6.5, x1=9.5,
        fillcolor="rgba(59, 130, 246, 0.1)",
        line_width=0,
        annotation_text="출근", annotation_position="top left",
        annotation_font_size=11, annotation_font_color="#3b82f6",
    )

    # 퇴근 피크 음영
    fig.add_vrect(
        x0=17.5, x1=20.5,
        fillcolor="rgba(245, 158, 11, 0.1)",
        line_width=0,
        annotation_text="퇴근", annotation_position="top left",
        annotation_font_size=11, annotation_font_color="#f59e0b",
    )
# ... [이하 생략 - 나머지 탭2, 탭3 구현은 동일하게 유지되며, 탭4 및 신설 탭5를 아래에 명시합니다] ...

    # 선택 시간 수직선
    fig.add_vline(x=selected_hour, line_dash="dot", line_color="#6366f1", line_width=2,
                  annotation_text=f"선택: {selected_hour}시", annotation_position="top right",
                  annotation_font_color="#6366f1")

    # 승차 라인
    fig.add_trace(go.Scatter(
        x=hours, y=boards,
        mode="lines+markers",
        name="승차인원",
        line=dict(color="#3b82f6", width=2.5),
        marker=dict(
            size=[10 if h == selected_hour else 6 for h in hours],
            color=["#6366f1" if h == selected_hour else "#3b82f6" for h in hours],
        ),
        hovertemplate="%{x}시 승차 %{y:,}명<extra></extra>",
    ))

    # 하차 라인
    fig.add_trace(go.Scatter(
        x=hours, y=alights,
        mode="lines+markers",
        name="하차인원",
        line=dict(color="#10b981", width=2.5),
        marker=dict(
            size=[10 if h == selected_hour else 6 for h in hours],
            color=["#059669" if h == selected_hour else "#10b981" for h in hours],
        ),
        hovertemplate="%{x}시 하차 %{y:,}명<extra></extra>",
    ))

    fig.update_layout(
        title=f"{selected_station}역 시간대별 승하차 예측 ({selected_date})",
        xaxis=dict(title="시간", tickmode="linear", tick0=5, dtick=1, ticksuffix="시"),
        yaxis=dict(title="인원 수 (명)", tickformat=","),
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

    # 피크 요약
    max_board_hr = hours[np.argmax(boards)]
    max_board    = max(boards)
    max_alight_hr = hours[np.argmax(alights)]
    max_alight   = max(alights)

    c1, c2 = st.columns(2)
    with c1:
        st.info(f"🔵 **승차 피크**: {max_board_hr}시 ({max_board:,}명)")
    with c2:
        st.success(f"🟢 **하차 피크**: {max_alight_hr}시 ({max_alight:,}명)")

# ================================================================================
# TAB 2: Weather Impact Simulator
# ================================================================================
with tab2:
    st.subheader(f"🌦 {selected_station}역 날씨 조건별 비교")
    st.caption(f"{selected_date} · {selected_hour}시 · 기온 {temp}°C 기준")

    # 3가지 조건 예측
    b_clear, a_clear = predict(selected_station, dt, selected_hour, temp, 0.0, 0.0)
    b_rain,  a_rain  = predict(selected_station, dt, selected_hour, temp, 10.0, 0.0)
    b_snow,  a_snow  = predict(selected_station, dt, selected_hour, temp-5, 0.0, 5.0)

    # 카드
    c1, c2, c3 = st.columns(3)
    conditions = [
        ("☀️ 맑음",  b_clear, a_clear, "#fef9c3", "#854d0e", "0mm 강수"),
        ("🌧 비",    b_rain,  a_rain,  "#dbeafe", "#1e40af", "10mm 강수"),
        ("❄️ 눈",    b_snow,  a_snow,  "#e0f2fe", "#0c4a6e", "5mm 적설"),
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

    # 비교 그래프 (하루 전체)
    st.subheader("하루 전체 날씨 조건별 승차 비교")
    hours_s, b_c, _ = predict_day(selected_station, dt, temp, 0.0, 0.0)
    _,        b_r, _ = predict_day(selected_station, dt, temp, 10.0, 0.0)
    _,        b_n, _ = predict_day(selected_station, dt, temp-5, 0.0, 5.0)

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=hours_s, y=b_c, name="맑음",  line=dict(color="#f59e0b", width=2), mode="lines"))
    fig2.add_trace(go.Scatter(x=hours_s, y=b_r, name="비",    line=dict(color="#3b82f6", width=2), mode="lines"))
    fig2.add_trace(go.Scatter(x=hours_s, y=b_n, name="눈",    line=dict(color="#06b6d4", width=2, dash="dot"), mode="lines"))

    # 차이 표시
    max_diff_rain = max([c-r for c,r in zip(b_c, b_r)])
    max_diff_snow = max([c-n for c,n in zip(b_c, b_n)])

    fig2.update_layout(
        title=f"맑음 vs 비 vs 눈 — 최대 차이: 비 -{max_diff_rain:,}명 / 눈 -{max_diff_snow:,}명",
        xaxis=dict(title="시간", tickmode="linear", tick0=5, dtick=1, ticksuffix="시"),
        yaxis=dict(title="승차인원 (명)", tickformat=","),
        plot_bgcolor="white", paper_bgcolor="white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=380,
        margin=dict(l=0, r=0, t=50, b=0),
    )
    fig2.update_xaxes(showgrid=True, gridcolor="#f1f5f9")
    fig2.update_yaxes(showgrid=True, gridcolor="#f1f5f9")
    st.plotly_chart(fig2, use_container_width=True)

    col_a, col_b = st.columns(2)
    rain_diff = round(np.mean([c-r for c,r in zip(b_c, b_r)]))
    snow_diff = round(np.mean([c-n for c,n in zip(b_c, b_n)]))
    col_a.warning(f"🌧 비 올 때 평균 {rain_diff:,}명 감소")
    col_b.info(f"❄️ 눈 올 때 평균 {snow_diff:,}명 감소")

# ================================================================================
# TAB 3: Station Congestion Rankings
# ================================================================================
with tab3:
    st.subheader(f"🏆 2호선 전체 역 혼잡도 랭킹")
    st.caption(f"{selected_date} · {selected_hour}시 · 기온 {temp}°C · 강수량 {rain}mm")

    with st.spinner("2호선 전체 역 예측 중..."):
        rankings = []
        for stn in STATIONS:
            b, a = predict(stn, dt, selected_hour, temp, rain, snow)
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

    # 전체 랭킹 바 차트
    fig3 = go.Figure()
    colors = []
    for _, row in rank_df.iterrows():
        _, ccol, _ = get_congestion(row["승차"], row["역명"])
        colors.append(ccol)

    fig3.add_trace(go.Bar(
        x=rank_df["역명"],
        y=rank_df["승차"],
        marker_color=colors,
        hovertemplate="%{x} 승차 %{y:,}명<extra></extra>",
        name="승차인원",
    ))
    fig3.update_layout(
        title=f"2호선 전체 역 승차인원 ({selected_hour}시)",
        xaxis=dict(title="역명", tickangle=-45, tickfont=dict(size=10)),
        yaxis=dict(title="승차인원 (명)", tickformat=","),
        plot_bgcolor="white", paper_bgcolor="white",
        height=380,
        margin=dict(l=0, r=0, t=50, b=80),
        showlegend=False,
    )
    fig3.update_xaxes(showgrid=False)
    fig3.update_yaxes(showgrid=True, gridcolor="#f1f5f9")
    st.plotly_chart(fig3, use_container_width=True)

# ================================================================================
# TAB 4: Smart Travel Planner (Optimal Departure Time)
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
                b_f, _ = predict(from_station, dt, hr, temp, rain, snow, active_model)
                b_t, _ = predict(to_station,   dt, hr, temp, rain, snow, active_model)
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

        # 현재 선택 시간 비교
        cur = res_df[res_df["시간"] == selected_hour].iloc[0]
        diff = int(cur["합산"]) - int(best["합산"])
        if diff > 0:
            st.info(f"ℹ️ 현재 선택한 {selected_hour}시보다 **{int(best['시간'])}시**에 타면 합산 **{diff:,}명** 더 한산해요.")

        # 비교 그래프
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
            title="시간대별 출발·도착역 혼잡도 비교",
            barmode="group",
            xaxis=dict(title="시간", tickmode="linear", tick0=5, dtick=1, ticksuffix="시"),
            yaxis=dict(title="승차인원 (명)", tickformat=","),
            plot_bgcolor="white", paper_bgcolor="white",
            height=380,
            margin=dict(l=0, r=0, t=50, b=0),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        fig4.update_xaxes(showgrid=False)
        fig4.update_yaxes(showgrid=True, gridcolor="#f1f5f9")
        st.plotly_chart(fig4, use_container_width=True)

    st.divider()
    st.caption("💡 최적 탑승 시간은 출발역 + 도착역 혼잡도 합산 기준으로 계산됩니다.")

# ================================================================================
# TAB 5: AI Model Benchmark & Comparison
# ================================================================================
with tab5:
    st.subheader("🤖 개발 모델별 실시간 예측 결과 비교 분석")
    st.markdown("팀원들이 개발한 4개의 예측 모델(XGBoost, LightGBM, RandomForest, LSTM)의 예측 수치와 혼잡도 수준을 대조하고 차이를 시각화합니다.")
    
    col_comp_info, col_comp_metric = st.columns([3, 2])
    with col_comp_info:
        st.info(f"🚉 비교 대상역: **{selected_station}역** · **{selected_hour}시**기준")
    
    # Evaluate only successfully loaded models
    model_list = [m_name for m_name, info in ALL_MODELS.items() if info.get("loaded", False)]
    comp_results = []
    
    for m_name in model_list:
        b_val, a_val = predict(selected_station, dt, selected_hour, temp, rain, snow, model_name=m_name)
        c_lbl, c_col, _ = get_congestion(b_val, selected_station)
        
        comp_results.append({
            "모델": m_name,
            "예상 승차 (명)": b_val,
            "예상 하차 (명)": a_val,
            "혼잡도 판정": f"<span style='color:{c_col}; font-weight:700;'>{c_lbl}</span>"
        })
        
    comp_df = pd.DataFrame(comp_results)
    
    with col_comp_info:
        # Render HTML table to preserve colors
        st.write(comp_df.to_html(escape=False, index=False), unsafe_allow_html=True)
        
    # Render 24h charts for active models
    st.markdown("---")
    
    col_chart1, col_chart2 = st.columns(2)
    
    # Model line color mappings
    model_colors = {
        "XGBoost": "#3b82f6",     # Blue
        "LightGBM": "#10b981",    # Green
        "RandomForest": "#8b5cf6", # Purple
        "LSTM": "#f43f5e"         # Ruby Red
    }
    
    with col_chart1:
        st.subheader("📈 모델별 24시간 승차 예측 패턴")
        fig_comp_board = go.Figure()
        for m_name in model_list:
            hrs, b_vals, _ = predict_day(selected_station, dt, temp, rain, snow, model_name=m_name)
            
            fig_comp_board.add_trace(go.Scatter(
                x=hrs, y=b_vals,
                mode="lines+markers",
                name=m_name,
                line=dict(color=model_colors[m_name], width=2.5),
                marker=dict(size=5),
                hovertemplate="%{x}시 %{y:,}명 승차<extra></extra>"
            ))
            
        fig_comp_board.update_layout(
            xaxis=dict(title="시간 (시)", tickmode="linear", tick0=5, dtick=2, ticksuffix="시"),
            yaxis=dict(title="예상 승차 인원 (명)", tickformat=","),
            plot_bgcolor="white", paper_bgcolor="white",
            height=400,
            margin=dict(l=0, r=0, t=10, b=0),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        fig_comp_board.update_xaxes(showgrid=False)
        fig_comp_board.update_yaxes(showgrid=True, gridcolor="#f1f5f9")
        st.plotly_chart(fig_comp_board, use_container_width=True)
        
    with col_chart2:
        st.subheader("📉 모델별 24시간 하차 예측 패턴")
        fig_comp_alight = go.Figure()
        for m_name in model_list:
            hrs, _, a_vals = predict_day(selected_station, dt, temp, rain, snow, model_name=m_name)
            
            fig_comp_alight.add_trace(go.Scatter(
                x=hrs, y=a_vals,
                mode="lines+markers",
                name=m_name,
                line=dict(color=model_colors[m_name], width=2.5),
                marker=dict(size=5),
                hovertemplate="%{x}시 %{y:,}명 하차<extra></extra>"
            ))
            
        fig_comp_alight.update_layout(
            xaxis=dict(title="시간 (시)", tickmode="linear", tick0=5, dtick=2, ticksuffix="시"),
            yaxis=dict(title="예상 하차 인원 (명)", tickformat=","),
            plot_bgcolor="white", paper_bgcolor="white",
            height=400,
            margin=dict(l=0, r=0, t=10, b=0),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        fig_comp_alight.update_xaxes(showgrid=False)
        fig_comp_alight.update_yaxes(showgrid=True, gridcolor="#f1f5f9")
        st.plotly_chart(fig_comp_alight, use_container_width=True)
        
    st.divider()
    st.caption("💡 현재 정상적으로 로드되어 가동 중인 2호선 승하차 예측 AI 모델들의 실시간 예측 수치를 비교 대조합니다.")

import os
import pandas as pd
import numpy as np
import joblib
import keras
import streamlit as st
from pathlib import Path
from core.config import ALL_LINE2_STATIONS

BASE_DIR = Path(__file__).resolve().parents[1]

# ── Keras 3 호환용 패치 클래스 선언 ──────────────────────────────────────────
@keras.saving.register_keras_serializable(package="Custom")
class PatchedEmbedding(keras.layers.Embedding):
    def __init__(self, *args, **kwargs):
        kwargs.pop('quantization_config', None)
        super().__init__(*args, **kwargs)

# ── LSTM 예측 모델 학습/추론용 과거 시계열 기준 데이터셋 로드 ──────────────────────────
@st.cache_data
def load_lstm_base_dataset():
    path = BASE_DIR / "data" / "processed" / "final_dataset_line2_230101-241231.csv"
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path)
        df['날짜'] = pd.to_datetime(df['날짜'])
        df['요일'] = df['날짜'].dt.weekday
        df['요일_시간'] = df['요일'] * 24 + df['시간']
        
        df = df.sort_values(by=['역명', '날짜', '시간']).reset_index(drop=True)
        
        if '역별_평균_승차' not in df.columns:
            stn_in = df.groupby('역명')['승차인원'].mean().to_dict()
            df['역별_평균_승차'] = df['역명'].map(stn_in)
        if '역별_평균_하차' not in df.columns:
            stn_out = df.groupby('역명')['하차인원'].mean().to_dict()
            df['역별_평균_하차'] = df['역명'].map(stn_out)
            
        for col in ['승차인원', '하차인원', '역별_평균_승차', '역별_평균_하차']:
            df[col] = np.log1p(df[col])
            
        return df[['역명', '시간', '승차인원', '하차인원', '요일', '월', '공휴일여부', '기온', '강수량', '적설', '요일_시간', '역별_평균_승차', '역별_평균_하차']]
    except Exception:
        return None

# ── 예측 AI 모델 로딩 영역 (XGBoost, LightGBM, RandomForest, LSTM) ─────────────────────
@st.cache_resource
def load_all_models():
    """
    4대 예측 AI 모델(XGBoost, LightGBM, RandomForest, LSTM) 파일 및 전처리용 인코더/스케일러를 로드합니다.
    """
    models = {}
    le_station = None
    
    try:
        le_station = joblib.load(BASE_DIR / "models" / "xgboost" / "label_encoder_station_line2.pkl")
    except Exception as e:
        print(f"[경고] 공통 label_encoder_station_line2.pkl 로드 실패 (XGBoost/LightGBM 사용 불가): {e}")

    # 1. XGBoost
    try:
        models["XGBoost"] = {
            "board": joblib.load(BASE_DIR / "models" / "xgboost" / "xgb_board_model_line2.pkl"),
            "alight": joblib.load(BASE_DIR / "models" / "xgboost" / "xgb_alight_model_line2.pkl"),
            "loaded": True
        }
    except Exception as e:
        print(f"[경고] XGBoost 모델 가중치 파일 로드 실패: {e}")
        models["XGBoost"] = {"loaded": False}

    # 2. LightGBM
    try:
        models["LightGBM"] = {
            "board": joblib.load(BASE_DIR / "models" / "lightgbm" / "lgb_boadin_model_line2.pkl"),
            "alight": joblib.load(BASE_DIR / "models" / "lightgbm" / "lgb_alight_model_line2.pkl"),
            "loaded": True
        }
    except Exception as e:
        print(f"[경고] LightGBM 모델 가중치 파일 로드 실패: {e}")
        models["LightGBM"] = {"loaded": False}

    # 3. RandomForest
    try:
        models["RandomForest"] = {
            "board": joblib.load(BASE_DIR / "models" / "randomforest" / "randomforest_boarding_model_line2.pkl"),
            "alight": joblib.load(BASE_DIR / "models" / "randomforest" / "randomforest_dropoff_model_line2.pkl"),
            "cols": joblib.load(BASE_DIR / "models" / "randomforest" / "model_boardin_columns_line2.pkl"),
            "loaded": True
        }
    except Exception as e:
        print(f"[경고] RandomForest 모델 가중치 파일 로드 실패: {e}")
        models["RandomForest"] = {"loaded": False}

    # 4. LSTM
    try:
        m_board = keras.models.load_model(
            BASE_DIR / "models" / "lstm" / "lstm_boarding_line2.keras",
            custom_objects={'Embedding': PatchedEmbedding}, 
            compile=False
        )
        m_alight = keras.models.load_model(
            BASE_DIR / "models" / "lstm" / "lstm_alighting_line2.keras",
            custom_objects={'Embedding': PatchedEmbedding}, 
            compile=False
        )
        
        try:
            m_board_jamsil = keras.models.load_model(
                BASE_DIR / "models" / "lstm" / "lstm_boarding_잠실_line2.keras",
                custom_objects={'Embedding': PatchedEmbedding}, 
                compile=False
            )
            m_alight_jamsil = keras.models.load_model(
                BASE_DIR / "models" / "lstm" / "lstm_alighting_잠실_line2.keras",
                custom_objects={'Embedding': PatchedEmbedding}, 
                compile=False
            )
        except Exception:
            m_board_jamsil = m_board
            m_alight_jamsil = m_alight

        scaler_path = BASE_DIR / "models" / "lstm" / "scaler_line2.pkl"
        if not os.path.exists(scaler_path):
            scaler_path = BASE_DIR / "models" / "lstm" / "scaler_x_line2.pkl"
            
        scaler = joblib.load(scaler_path)
        le_lstm = joblib.load(BASE_DIR / "models" / "lstm" / "label_encoder_line2.pkl")

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

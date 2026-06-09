import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from core.numpy_lstm import NumpyLSTMModel

BASE_DIR = Path(__file__).resolve().parents[1]


@st.cache_data
def load_lstm_base_dataset():
    path = BASE_DIR / "data" / "processed" / "final_dataset_line1_8_230101-241231.csv"
    if not os.path.exists(path):
        return None

    try:
        df = pd.read_csv(path)
        df["날짜"] = pd.to_datetime(df["날짜"])
        df["요일"] = df["날짜"].dt.weekday
        df["요일_시간"] = df["요일"] * 24 + df["시간"]
        df["역명"] = df["역명"].astype(str) + "_" + df["호선"].astype(str)

        df = df.sort_values(by=["역명", "날짜", "시간"]).reset_index(drop=True)

        if "역별_평균_승차" not in df.columns:
            stn_in = df.groupby("역명")["승차인원"].mean().to_dict()
            df["역별_평균_승차"] = df["역명"].map(stn_in)
        if "역별_평균_하차" not in df.columns:
            stn_out = df.groupby("역명")["하차인원"].mean().to_dict()
            df["역별_평균_하차"] = df["역명"].map(stn_out)

        for col in ["승차인원", "하차인원", "역별_평균_승차", "역별_평균_하차"]:
            df[col] = np.log1p(df[col])

        return df[
            [
                "역명",
                "시간",
                "승차인원",
                "하차인원",
                "요일",
                "월",
                "공휴일여부",
                "기온",
                "강수량",
                "적설",
                "요일_시간",
                "역별_평균_승차",
                "역별_평균_하차",
            ]
        ]
    except Exception:
        return None


@st.cache_data
def load_station_avg_map():
    df = load_lstm_base_dataset()
    if df is None or df.empty:
        return {}

    raw_board = np.expm1(df["승차인원"])
    raw_df = pd.DataFrame({"역명": df["역명"], "승차인원_raw": raw_board})
    return raw_df.groupby("역명")["승차인원_raw"].mean().to_dict()


@st.cache_resource
def load_all_models():
    models = {}
    le_station = None
    le_line = None

    try:
        le_station = joblib.load(BASE_DIR / "models" / "lstm" / "label_encoder_station_line1_8.pkl")
        le_line = joblib.load(BASE_DIR / "models" / "lstm" / "label_encoder_line_line1_8.pkl")
    except Exception as e:
        print(f"[경고] 1-8호선 label encoder 로드 실패: {e}")

    try:
        if le_station is None or le_line is None:
            raise RuntimeError("LSTM label encoders are not loaded.")

        board_model = NumpyLSTMModel(BASE_DIR / "models" / "lstm" / "lstm_boarding_line1_8.keras")
        alight_model = NumpyLSTMModel(BASE_DIR / "models" / "lstm" / "lstm_alighting_line1_8.keras")
        scaler = joblib.load(BASE_DIR / "models" / "lstm" / "scaler_line1_8.pkl")
        le_lstm = joblib.load(BASE_DIR / "models" / "lstm" / "label_encoder_station_line1_8.pkl")

        models["LSTM"] = {
            "board": board_model,
            "alight": alight_model,
            "scaler": scaler,
            "le": le_lstm,
            "le_line": le_line,
            "station_avg_map": load_station_avg_map(),
            "backend": "numpy",
            "loaded": True,
        }
    except Exception as e:
        print(f"[경고] 1-8호선 LSTM 모델 로드 실패: {e}")
        models["LSTM"] = {"loaded": False}

    try:
        board_pack = joblib.load(BASE_DIR / "models" / "lightgbm" / "lgb_boadin_model_line1_8.pkl")
        alight_pack = joblib.load(BASE_DIR / "models" / "lightgbm" / "lgb_alight_model_line1_8.pkl")

        models["LightGBM"] = {
            "board": board_pack["model"],
            "alight": alight_pack["model"],
            "columns": list(board_pack.get("columns", board_pack["model"].feature_name_)),
            "loaded": True,
        }
    except Exception as e:
        print(f"[경고] 1-8호선 LightGBM 모델 로드 실패: {e}")
        models["LightGBM"] = {"loaded": False}

    try:
        board_model = joblib.load(BASE_DIR / "models" / "xgboost" / "xgb_board_model_line1_8.pkl")
        alight_model = joblib.load(BASE_DIR / "models" / "xgboost" / "xgb_alight_model_line1_8.pkl")
        le_xgb_station = joblib.load(BASE_DIR / "models" / "xgboost" / "label_encoder_station_line1_8.pkl")
        le_xgb_line = joblib.load(BASE_DIR / "models" / "xgboost" / "label_encoder_line_line1_8.pkl")

        models["XGBoost"] = {
            "board": board_model,
            "alight": alight_model,
            "le_station": le_xgb_station,
            "le_line": le_xgb_line,
            "columns": list(getattr(board_model, "feature_names_in_", [])),
            "loaded": True,
        }
    except Exception as e:
        print(f"[경고] 1-8호선 XGBoost 모델 로드 실패: {e}")
        models["XGBoost"] = {"loaded": False}

    return models, le_station, any(info.get("loaded", False) for info in models.values())

import json
import io
import os
import zipfile
from datetime import date, datetime, timedelta
from pathlib import Path

import h5py
import holidays as hd
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from services.subway_api import check_holiday
from services.weather_api import get_weather_with_fallback


BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
LINE2_DATASET = PROJECT_ROOT / "data" / "processed" / "final_dataset_line2_230101-241231.csv"

MAIN_LINE = [
    "시청", "을지로입구", "을지로3가", "을지로4가", "동대문역사문화공원(DDP)",
    "신당", "상왕십리", "왕십리", "한양대", "뚝섬", "성수", "건대입구",
    "구의", "강변", "잠실나루", "잠실", "잠실새내", "종합운동장", "삼성",
    "선릉", "역삼", "강남", "교대", "서초", "방배", "사당", "낙성대",
    "서울대입구", "봉천", "신림", "신대방", "구로디지털단지", "대림",
    "신도림", "문래", "영등포구청", "당산", "합정", "홍대입구", "신촌",
    "이대", "아현", "충정로",
]
SUNGSU_BRANCH = ["용답", "신답", "용두(동대문구청)", "신설동"]
SINDORIM_BRANCH = ["도림천", "양천구청", "신정네거리", "까치산"]
ALL_LINE2_STATIONS = MAIN_LINE + SUNGSU_BRANCH + SINDORIM_BRANCH
LINE2_COLOR = "#00A84D"

LINE2_FEATURE_COLS = [
    "역명_enc",
    "시간", "시간_sin", "시간_cos",
    "요일", "요일_sin", "요일_cos",
    "월", "월_sin", "월_cos",
    "공휴일여부", "비근무일",
    "출근피크", "퇴근피크",
    "기온", "강수량", "적설",
    "강수_여부", "적설_여부", "불쾌지수",
    "공휴일_유형", "연휴_여부", "공휴일_전날", "공휴일_다음날",
]

LINE2_STATION_AVG = {station: 1500 for station in ALL_LINE2_STATIONS}
LINE2_STATION_AVG.update({
    "강남": 3200, "홍대입구": 2800, "잠실": 2600, "신림": 2200,
    "신도림": 2400, "건대입구": 2000, "사당": 2100, "왕십리": 1800,
    "선릉": 1900, "역삼": 1800, "교대": 1600, "합정": 1500,
    "시청": 1200, "을지로입구": 1400, "신설동": 1100, "용두(동대문구청)": 600,
    "신답": 400, "용답": 700, "까치산": 1800, "신정네거리": 1200,
    "양천구청": 900, "도림천": 300,
})


def load_line2_env():
    for env_path in [PROJECT_ROOT / ".env", BASE_DIR / ".env"]:
        if not env_path.exists():
            continue
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    os.environ.setdefault(key.strip(), value.strip())


@st.cache_data(show_spinner=False)
def load_line2_transfer_info():
    path = PROJECT_ROOT / "data" / "transfer_info.json"
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


@st.cache_data(show_spinner=False)
def load_line2_base_dataset():
    if not LINE2_DATASET.exists():
        return pd.DataFrame()

    try:
        df = pd.read_csv(LINE2_DATASET)
        df["날짜"] = pd.to_datetime(df["날짜"])
        df["요일"] = df["날짜"].dt.weekday
        df["요일_시간"] = df["요일"] * 24 + df["시간"]
        df = df.sort_values(by=["역명", "날짜", "시간"]).reset_index(drop=True)

        if "역별_평균_승차" not in df.columns:
            df["역별_평균_승차"] = df["역명"].map(df.groupby("역명")["승차인원"].mean())
        if "역별_평균_하차" not in df.columns:
            df["역별_평균_하차"] = df["역명"].map(df.groupby("역명")["하차인원"].mean())

        for col in ["승차인원", "하차인원", "역별_평균_승차", "역별_평균_하차"]:
            df[col] = np.log1p(df[col])

        return df[
            [
                "역명", "시간", "승차인원", "하차인원", "요일", "월", "공휴일여부",
                "기온", "강수량", "적설", "요일_시간", "역별_평균_승차", "역별_평균_하차",
            ]
        ]
    except Exception:
        return pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_line2_metric_sample(station, sample_size=48):
    if not LINE2_DATASET.exists():
        return pd.DataFrame()
    df = pd.read_csv(
        LINE2_DATASET,
        usecols=["날짜", "역명", "시간", "승차인원", "하차인원", "기온", "강수량", "적설"],
        parse_dates=["날짜"],
    )
    matched = df[df["역명"].astype(str) == station].copy()
    return matched.sort_values(["날짜", "시간"]).tail(sample_size).reset_index(drop=True)


@st.cache_data(show_spinner=False)
def load_line2_validation_sample(seed=42, sample_size=100):
    if not LINE2_DATASET.exists():
        return pd.DataFrame()

    df = pd.read_csv(
        LINE2_DATASET,
        usecols=["날짜", "역명", "시간", "승차인원", "하차인원", "기온", "강수량", "적설"],
        parse_dates=["날짜"],
    )
    recent_df = df[df["날짜"] >= "2024-12-15"].copy()
    if recent_df.empty:
        return recent_df

    return recent_df.sample(min(sample_size, len(recent_df)), random_state=seed).reset_index(drop=True)


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60, 60)))


def _relu(x):
    return np.maximum(x, 0.0)


class NumpyLine2LSTMModel:
    def __init__(self, model_path):
        self.model_path = model_path
        self.weights = self._load_weights(model_path)

    @staticmethod
    def _read_dataset(group, path):
        return np.asarray(group[path], dtype=np.float32)

    @classmethod
    def _load_weights(cls, model_path):
        with zipfile.ZipFile(model_path) as zf:
            weights_bytes = zf.read("model.weights.h5")

        with h5py.File(io.BytesIO(weights_bytes), "r") as h5:
            return {
                "lstm0_kernel": cls._read_dataset(h5, "layers/lstm/cell/vars/0"),
                "lstm0_recurrent": cls._read_dataset(h5, "layers/lstm/cell/vars/1"),
                "lstm0_bias": cls._read_dataset(h5, "layers/lstm/cell/vars/2"),
                "lstm1_kernel": cls._read_dataset(h5, "layers/lstm_1/cell/vars/0"),
                "lstm1_recurrent": cls._read_dataset(h5, "layers/lstm_1/cell/vars/1"),
                "lstm1_bias": cls._read_dataset(h5, "layers/lstm_1/cell/vars/2"),
                "station_embedding": cls._read_dataset(h5, "layers/embedding/vars/0"),
                "dense0_w": cls._read_dataset(h5, "layers/dense/vars/0"),
                "dense0_b": cls._read_dataset(h5, "layers/dense/vars/1"),
                "dense1_w": cls._read_dataset(h5, "layers/dense_1/vars/0"),
                "dense1_b": cls._read_dataset(h5, "layers/dense_1/vars/1"),
            }

    @staticmethod
    def _lstm(x, kernel, recurrent_kernel, bias, return_sequences):
        x = np.asarray(x, dtype=np.float32)
        batch, steps, _ = x.shape
        units = recurrent_kernel.shape[0]
        h = np.zeros((batch, units), dtype=np.float32)
        c = np.zeros((batch, units), dtype=np.float32)
        outputs = []

        for t in range(steps):
            z = x[:, t, :] @ kernel + h @ recurrent_kernel + bias
            i = _sigmoid(z[:, :units])
            f = _sigmoid(z[:, units:2 * units])
            c_bar = np.tanh(z[:, 2 * units:3 * units])
            o = _sigmoid(z[:, 3 * units:])
            c = f * c + i * c_bar
            h = o * np.tanh(c)
            outputs.append(h)

        if return_sequences:
            return np.stack(outputs, axis=1)
        return h

    def __call__(self, inputs, training=False):
        del training

        seq = np.asarray(inputs[0], dtype=np.float32)
        station_idx = np.asarray(inputs[1]).astype(int).reshape(-1)
        w = self.weights

        x = self._lstm(seq, w["lstm0_kernel"], w["lstm0_recurrent"], w["lstm0_bias"], return_sequences=True)
        x = self._lstm(x, w["lstm1_kernel"], w["lstm1_recurrent"], w["lstm1_bias"], return_sequences=False)
        station_emb = w["station_embedding"][station_idx]
        x = np.concatenate([x, station_emb], axis=1)
        x = _relu(x @ w["dense0_w"] + w["dense0_b"])
        return x @ w["dense1_w"] + w["dense1_b"]


@st.cache_resource(show_spinner="2호선 모델을 불러오는 중...")
def load_line2_models():
    models = {}
    le_station = None

    try:
        le_station = joblib.load(PROJECT_ROOT / "models" / "xgboost" / "label_encoder_station_line2.pkl")
    except Exception as e:
        print(f"[경고] 2호선 label encoder 로드 실패: {e}")

    try:
        models["XGBoost"] = {
            "board": joblib.load(PROJECT_ROOT / "models" / "xgboost" / "xgb_board_model_line2.pkl"),
            "alight": joblib.load(PROJECT_ROOT / "models" / "xgboost" / "xgb_alight_model_line2.pkl"),
            "loaded": True,
        }
    except Exception as e:
        print(f"[경고] 2호선 XGBoost 로드 실패: {e}")
        models["XGBoost"] = {"loaded": False}

    try:
        models["LightGBM"] = {
            "board": joblib.load(PROJECT_ROOT / "models" / "lightgbm" / "lgb_boadin_model_line2.pkl"),
            "alight": joblib.load(PROJECT_ROOT / "models" / "lightgbm" / "lgb_alight_model_line2.pkl"),
            "loaded": True,
        }
    except Exception as e:
        print(f"[경고] 2호선 LightGBM 로드 실패: {e}")
        models["LightGBM"] = {"loaded": False}

    try:
        models["RandomForest"] = {
            "board": joblib.load(PROJECT_ROOT / "models" / "randomforest" / "randomforest_boarding_model_line2.pkl"),
            "alight": joblib.load(PROJECT_ROOT / "models" / "randomforest" / "randomforest_dropoff_model_line2.pkl"),
            "cols": joblib.load(PROJECT_ROOT / "models" / "randomforest" / "model_boardin_columns_line2.pkl"),
            "loaded": True,
        }
    except Exception as e:
        print(f"[경고] 2호선 RandomForest 로드 실패: {e}")
        models["RandomForest"] = {"loaded": False}

    try:
        models["LSTM"] = {
            "board": NumpyLine2LSTMModel(PROJECT_ROOT / "models" / "lstm" / "lstm_boarding_line2.keras"),
            "alight": NumpyLine2LSTMModel(PROJECT_ROOT / "models" / "lstm" / "lstm_alighting_line2.keras"),
            "scaler": joblib.load(PROJECT_ROOT / "models" / "lstm" / "scaler_line2.pkl"),
            "le": joblib.load(PROJECT_ROOT / "models" / "lstm" / "label_encoder_line2.pkl"),
            "backend": "numpy",
            "loaded": True,
        }
    except Exception as e:
        print(f"[경고] 2호선 LSTM 로드 실패: {e}")
        models["LSTM"] = {"loaded": False}

    return models, le_station, any(info.get("loaded", False) for info in models.values())


def get_line2_holiday_type(d, kr_hols):
    if d not in kr_hols:
        return 0
    name = kr_hols.get(d, "")
    if any(k in name for k in ["Korean New Year", "Chuseok", "설날", "추석", "preceding", "second day"]):
        return 3
    if any(k in name for k in ["Children", "Christmas", "Buddha", "어린이날", "성탄절", "기독탄신일", "부처님오신날", "석가탄신일"]):
        return 2
    return 1


def get_line2_avg(station, base_df=None, fallback=1500):
    if base_df is not None and not base_df.empty:
        matched = base_df[base_df["역명"] == station]
        if not matched.empty:
            return float(np.expm1(matched["승차인원"]).mean())
    return float(LINE2_STATION_AVG.get(station, fallback))


def build_line2_feature_row(station, dt, hour, temp, rain, snow, le_station):
    kr_hols = hd.KR(years=dt.year)
    is_hol = int(dt.date() in kr_hols)
    stn_enc = int(le_station.transform([station])[0]) if le_station is not None and station in le_station.classes_ else 0
    prev_d = dt - timedelta(days=1)
    next_d = dt + timedelta(days=1)
    prev_off = (prev_d.weekday() >= 5) or (prev_d.date() in kr_hols)
    next_off = (next_d.weekday() >= 5) or (next_d.date() in kr_hols)
    is_off = (dt.weekday() >= 5) or bool(is_hol)

    return {
        "역명_enc": stn_enc,
        "시간": hour,
        "시간_sin": np.sin(2 * np.pi * hour / 24),
        "시간_cos": np.cos(2 * np.pi * hour / 24),
        "요일": dt.weekday(),
        "요일_sin": np.sin(2 * np.pi * dt.weekday() / 7),
        "요일_cos": np.cos(2 * np.pi * dt.weekday() / 7),
        "월": dt.month,
        "월_sin": np.sin(2 * np.pi * dt.month / 12),
        "월_cos": np.cos(2 * np.pi * dt.month / 12),
        "공휴일여부": is_hol,
        "비근무일": int(is_off),
        "출근피크": int(hour in [7, 8, 9]),
        "퇴근피크": int(hour in [18, 19, 20]),
        "기온": temp,
        "강수량": rain,
        "적설": snow,
        "강수_여부": int(rain > 0),
        "적설_여부": int(snow > 0),
        "불쾌지수": round(9 / 5 * temp - 0.55 * (1 - int(rain > 0) * 0.8) * (9 / 5 * temp - 26) + 32, 2),
        "공휴일_유형": get_line2_holiday_type(dt.date(), kr_hols),
        "연휴_여부": int(is_off and (prev_off or next_off)),
        "공휴일_전날": int(next_d.date() in kr_hols),
        "공휴일_다음날": int(prev_d.date() in kr_hols),
    }


def line2_predict(station, dt, hour, temp, rain=0.0, snow=0.0, model_name="XGBoost", all_models=None, le_station=None, base_df=None):
    if all_models is None:
        raise ValueError("models must be provided.")

    m_info = all_models.get(model_name, {})
    if not m_info.get("loaded", False):
        raise ValueError(f"2호선 {model_name} 모델이 로드되지 않았습니다.")

    row = build_line2_feature_row(station, dt, hour, temp, rain, snow, le_station)
    X = pd.DataFrame([row])[LINE2_FEATURE_COLS]
    kr_hols = hd.KR(years=dt.year)
    is_hol = int(dt.date() in kr_hols)

    if model_name == "RandomForest":
        rf_cols = m_info["cols"]
        row_rf = {col: 0.0 for col in rf_cols}
        row_rf.update({
            "시간": float(hour),
            "요일": float(dt.weekday()),
            "월": float(dt.month),
            "공휴일여부": float(is_hol),
            "기온": float(temp),
            "강수량": float(rain),
            "적설": float(snow),
        })
        for col in rf_cols:
            if col.startswith("역명_") and col.replace("역명_", "") == station:
                row_rf[col] = 1.0
        X_model = pd.DataFrame([row_rf])[rf_cols]
    elif model_name == "LightGBM":
        station_categories = sorted(list(le_station.classes_)) if le_station is not None else sorted(ALL_LINE2_STATIONS)
        X_model = pd.DataFrame([{
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
            "weekday": int(dt.weekday()),
        }])
        X_model["역명"] = pd.Categorical(X_model["역명"], categories=station_categories)
        X_model["호선"] = pd.Categorical(X_model["호선"], categories=["2호선"])
        X_model = X_model[["역명", "호선", "시간", "요일", "월", "공휴일여부", "기온", "강수량", "적설", "year", "day", "weekday"]]
    elif model_name == "LSTM":
        try:
            scaler = m_info["scaler"]
            le_lstm = m_info.get("le")
            station_idx = 0
            if le_lstm is not None:
                for idx, cls in enumerate(le_lstm.classes_):
                    if station in cls or cls in station:
                        station_idx = idx
                        break

            avg_val = get_line2_avg(station, base_df)
            log_avg_val = np.log1p(avg_val)
            scaler_cols = [
                "승차인원", "하차인원", "시간", "요일", "월", "공휴일여부",
                "기온", "강수량", "적설", "요일_시간", "역별_평균_승차", "역별_평균_하차",
            ]
            row_scaler = {
                "승차인원": 0.0,
                "하차인원": 0.0,
                "시간": float(hour),
                "요일": float(dt.weekday()),
                "월": float(dt.month),
                "공휴일여부": float(is_hol),
                "기온": float(temp),
                "강수량": float(rain),
                "적설": float(snow),
                "요일_시간": float(dt.weekday() * 24 + hour),
                "역별_평균_승차": float(log_avg_val),
                "역별_평균_하차": float(log_avg_val),
            }
            feat_12 = list(scaler.transform(pd.DataFrame([row_scaler])[scaler_cols])[0])

            if base_df is not None and not base_df.empty:
                matched_station = station
                if le_lstm is not None and len(le_lstm.classes_) > station_idx:
                    matched_station = le_lstm.classes_[station_idx]
                station_all = base_df[base_df["역명"] == matched_station].copy().reset_index(drop=True)
                if station_all.empty:
                    seq = np.array([feat_12] * 12)
                else:
                    same_hour_idx = station_all[station_all["시간"] == hour].index.tolist()
                    if not same_hour_idx:
                        same_hour_idx = [int(np.abs(station_all["시간"] - hour).idxmin())]
                    best_idx = same_hour_idx[-1]
                    seq_raw = station_all.iloc[max(0, best_idx - 12):best_idx].tail(12)[scaler_cols].values.copy()
                    if len(seq_raw) < 12:
                        seq_raw = station_all.iloc[best_idx:best_idx + 12].tail(12)[scaler_cols].values.copy()
                    seq = scaler.transform(seq_raw) if len(seq_raw) else np.array([feat_12] * 12)
                    if len(seq) < 12:
                        seq = np.vstack([np.array([feat_12] * (12 - len(seq))), seq])
                    seq[-1] = feat_12
            else:
                seq = np.array([feat_12] * 12)

            X_in = np.expand_dims(seq, axis=0)
            stn_in = np.array([[float(station_idx)]])
            pred_b = np.array(m_info["board"]([X_in, stn_in], training=False))
            pred_a = np.array(m_info["alight"]([X_in, stn_in], training=False))

            dummy_b = np.zeros((1, 12))
            dummy_a = np.zeros((1, 12))
            dummy_b[0, 0] = pred_b.flatten()[0]
            dummy_a[0, 1] = pred_a.flatten()[0]
            b = int(np.clip(np.expm1(scaler.inverse_transform(dummy_b)[0, 0]), 0, None))
            a = int(np.clip(np.expm1(scaler.inverse_transform(dummy_a)[0, 1]), 0, None))
            return b, a
        except Exception:
            fallback = all_models.get("XGBoost", {})
            if fallback.get("loaded", False):
                return (
                    int(np.clip(fallback["board"].predict(X), 0, None)[0]),
                    int(np.clip(fallback["alight"].predict(X), 0, None)[0]),
                )
            raise
    else:
        X_model = X

    b = int(np.clip(m_info["board"].predict(X_model), 0, None)[0])
    a = int(np.clip(m_info["alight"].predict(X_model), 0, None)[0])
    return b, a


def line2_predict_day(station, dt, temp, rain, snow, model_name, all_models, le_station, base_df):
    hours, boards, alights = [], [], []
    for hour in range(5, 24):
        b, a = line2_predict(station, dt, hour, temp, rain, snow, model_name, all_models, le_station, base_df)
        hours.append(hour)
        boards.append(b)
        alights.append(a)
    return hours, boards, alights


def line2_get_congestion(board, station, avg=None):
    avg = avg if avg is not None else LINE2_STATION_AVG.get(station, 1500)
    ratio = board / avg if avg > 0 else 1
    if ratio < 0.6:
        return "쾌적", "#065f46", "#d1fae5"
    if ratio < 1.0:
        return "보통", "#1e40af", "#dbeafe"
    if ratio < 1.5:
        return "혼잡", "#92400e", "#fef3c7"
    return "매우혼잡", "#991b1b", "#fee2e2"


def render_line2_route_map(selected_station, transfers):
    station_coords = {}
    radius = 17.5
    for idx, name in enumerate(MAIN_LINE):
        theta = np.pi / 2 - (2 * np.pi * idx / len(MAIN_LINE))
        station_coords[name] = (radius * np.cos(theta), radius * np.sin(theta), "main")

    sx, sy, _ = station_coords["성수"]
    for idx, name in enumerate(SUNGSU_BRANCH):
        station_coords[name] = (sx + (idx + 1) * 1.4, sy + (idx + 1) * 2.1, "branch")

    dx, dy, _ = station_coords["신도림"]
    for idx, name in enumerate(SINDORIM_BRANCH):
        station_coords[name] = (dx - (idx + 1) * 1.7, dy - (idx + 1) * 1.7, "branch")

    fig = go.Figure()
    main_x = [station_coords[name][0] for name in MAIN_LINE] + [station_coords[MAIN_LINE[0]][0]]
    main_y = [station_coords[name][1] for name in MAIN_LINE] + [station_coords[MAIN_LINE[0]][1]]
    fig.add_trace(go.Scatter(x=main_x, y=main_y, mode="lines", line=dict(color=LINE2_COLOR, width=5), hoverinfo="skip", showlegend=False))

    for start, branch in [("성수", SUNGSU_BRANCH), ("신도림", SINDORIM_BRANCH)]:
        x_vals = [station_coords[start][0]] + [station_coords[name][0] for name in branch]
        y_vals = [station_coords[start][1]] + [station_coords[name][1] for name in branch]
        fig.add_trace(go.Scatter(x=x_vals, y=y_vals, mode="lines", line=dict(color=LINE2_COLOR, width=4, dash="dash"), hoverinfo="skip", showlegend=False))

    node_names = list(station_coords.keys())
    node_x = [station_coords[name][0] for name in node_names]
    node_y = [station_coords[name][1] for name in node_names]
    hub_names = {"강남", "잠실", "홍대입구", "신도림", "사당", "신림", "시청", "건대입구", "성수", "왕십리", "교대"}
    marker_sizes = []
    marker_colors = []
    marker_lines = []
    labels = []

    for name in node_names:
        is_selected = name == selected_station
        is_transfer = bool(transfers.get(name)) and any(transfers.get(name))
        marker_sizes.append(16 if is_selected else 10 if is_transfer else 7)
        marker_colors.append("#ffffff" if is_selected or is_transfer else LINE2_COLOR)
        marker_lines.append(4 if is_selected else 2)
        labels.append(name.split("(", 1)[0] if is_selected or is_transfer or name in hub_names else "")

    fig.add_trace(go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers+text",
        marker=dict(size=marker_sizes, color=marker_colors, line=dict(color=LINE2_COLOR, width=marker_lines)),
        text=labels,
        textposition="top center",
        textfont=dict(size=11, color="#111827"),
        customdata=node_names,
        hovertemplate="%{customdata}역<extra></extra>",
        showlegend=False,
    ))
    fig.update_layout(
        title="서울 지하철 2호선 노선도",
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        plot_bgcolor="white",
        paper_bgcolor="white",
        height=560,
        margin=dict(l=5, r=5, t=48, b=5),
    )
    st.plotly_chart(fig, use_container_width=True)


def calculate_mae(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_true - y_pred)))


def label_line2_validation_status(mae):
    if pd.isna(mae):
        return "계산 실패"
    if mae < 150:
        return "최우수"
    if mae < 200:
        return "우수"
    if mae < 300:
        return "보통"
    return "개선 필요"


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


def render_line2_dashboard():
    load_line2_env()
    transfers = load_line2_transfer_info()
    base_df = load_line2_base_dataset()
    all_models, le_station, loaded = load_line2_models()

    st.title("서울 2호선 실시간 혼잡도 예측")
    st.caption("기존 2호선 전용 모델과 데이터를 유지한 전용 대시보드입니다.")

    if not loaded:
        st.error("2호선 예측 모델을 찾지 못했습니다. 루트 `models/` 경로를 확인해 주세요.")
        return

    stations = sorted(list(le_station.classes_)) if le_station is not None else sorted(ALL_LINE2_STATIONS)
    loaded_models = [name for name, info in all_models.items() if info.get("loaded", False)]

    with st.sidebar:
        st.header("2호선 예측 설정")
        if "line2_model" not in st.session_state or st.session_state["line2_model"] not in loaded_models:
            st.session_state["line2_model"] = "XGBoost" if "XGBoost" in loaded_models else loaded_models[0]

        active_model = st.selectbox(
            "사용 모델",
            loaded_models,
            index=loaded_models.index(st.session_state["line2_model"]),
            key="line2_model",
        )

        default_station = "강남" if "강남" in stations else stations[0]
        if "line2_station" not in st.session_state or st.session_state["line2_station"] not in stations:
            st.session_state["line2_station"] = default_station

        selected_station = st.selectbox(
            "역 선택",
            stations,
            index=stations.index(st.session_state["line2_station"]),
            key="line2_station",
        )

        station_transfers = transfers.get(selected_station, [])
        if station_transfers and any(station_transfers):
            st.caption(f"환승: {', '.join([line for line in station_transfers if line])}")

        st.divider()
        selected_date = st.date_input(
            "날짜",
            value=date.today(),
            min_value=date(2020, 1, 1),
            max_value=date(2030, 12, 31),
            key="line2_date",
        )
        selected_hour = st.slider("시간", min_value=5, max_value=23, value=9, format="%d시", key="line2_hour")

        st.divider()
        st.subheader("날씨")
        weather_mode = st.radio("입력 방식", ["기상청 API (자동)", "수동 입력"], index=0, key="line2_weather_mode")

    if weather_mode == "기상청 API (자동)":
        api_key = os.environ.get("KMA_API_KEY", "")
        cache_key = (selected_date.isoformat(), selected_hour, "line2")
        if "line2_weather" not in st.session_state or st.session_state.get("line2_weather_key") != cache_key:
            with st.spinner("기상청 데이터를 불러오는 중..."):
                result = get_weather_with_fallback(api_key, selected_date, selected_hour)
                st.session_state["line2_weather"] = result[:3]
                st.session_state["line2_weather_source"] = result[3]
                st.session_state["line2_weather_key"] = cache_key

        temp, rain, snow = st.session_state["line2_weather"]
        source = st.session_state.get("line2_weather_source", "기본값")
        st.success(f"현재 날씨 ({source}): 기온 {temp}℃ / 강수량 {rain}mm / 적설 {snow}mm")

        if st.button("2호선 날씨 새로고침"):
            st.session_state.pop("line2_weather", None)
            st.rerun()
    else:
        temp = st.slider("기온 (℃)", min_value=-15.0, max_value=40.0, value=15.0, step=0.5, key="line2_temp")
        rain = st.slider("강수량 (mm)", min_value=0.0, max_value=100.0, value=0.0, step=0.5, key="line2_rain")
        snow = st.slider("적설 (mm)", min_value=0.0, max_value=50.0, value=0.0, step=0.5, key="line2_snow")

    dt = datetime.combine(selected_date, datetime.min.time())
    hol_name, hol_msg = check_holiday(dt)
    if hol_msg:
        icon = "공휴일" if hol_name else "일정"
        st.markdown(f'<div class="holiday-banner">{icon}: {hol_msg}</div>', unsafe_allow_html=True)

    tab0, tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "2호선 노선도",
        "시간대 예측",
        "날씨 비교",
        "역별 랭킹",
        "최적 탑승 시간",
        "모델별 예측치 비교",
    ])

    with tab0:
        render_line2_route_map(selected_station, transfers)

    with tab1:
        avg = get_line2_avg(selected_station, base_df)
        board_now, alight_now = line2_predict(selected_station, dt, selected_hour, temp, rain, snow, active_model, all_models, le_station, base_df)
        cong_label, cong_color, cong_bg = line2_get_congestion(board_now, selected_station, avg=avg)

        col_info, col_metric = st.columns([3, 2])
        with col_info:
            st.subheader(f"{selected_station}역")
            st.caption(f"{dt.strftime('%Y-%m-%d')} · {selected_hour}시 · 기온 {temp}℃ · 강수량 {rain}mm · 적설 {snow}mm")
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
        m1.metric("예측 승차인원", f"{board_now:,}명")
        m2.metric("예측 하차인원", f"{alight_now:,}명")
        diff_pct = round((board_now - avg) / avg * 100) if avg > 0 else 0
        m3.metric("역 평균 대비", f"{diff_pct:+}%", delta="평균 초과" if diff_pct > 0 else "평균 이하")

        hours, boards, alights = line2_predict_day(selected_station, dt, temp, rain, snow, active_model, all_models, le_station, base_df)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=hours, y=boards, mode="lines+markers", name="승차인원", line=dict(color="#3b82f6", width=2.5)))
        fig.add_trace(go.Scatter(x=hours, y=alights, mode="lines+markers", name="하차인원", line=dict(color="#10b981", width=2.5)))
        fig.add_vline(x=selected_hour, line_dash="dot", line_color="#6366f1", annotation_text=f"{selected_hour}시")
        fig.update_layout(
            title=f"{selected_station}역 시간대별 예측",
            xaxis=dict(title="시간", tickmode="linear", tick0=5, dtick=1, ticksuffix="시"),
            yaxis=dict(title="인원 (명)", tickformat=","),
            hovermode="x unified",
            plot_bgcolor="white",
            paper_bgcolor="white",
            height=420,
            margin=dict(l=0, r=0, t=50, b=0),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        fig.update_yaxes(showgrid=True, gridcolor="#f1f5f9")
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.subheader("날씨 조건별 비교")
        scenarios = [
            ("맑음", temp, 0.0, 0.0, "#f59e0b"),
            ("비", temp, 10.0, 0.0, "#3b82f6"),
            ("눈", temp - 5, 0.0, 5.0, "#06b6d4"),
        ]
        cols = st.columns(3)
        for col, (label, s_temp, s_rain, s_snow, color) in zip(cols, scenarios):
            b, a = line2_predict(selected_station, dt, selected_hour, s_temp, s_rain, s_snow, active_model, all_models, le_station, base_df)
            col.metric(label, f"{b:,}명 승차", f"{a:,}명 하차")

        fig_weather = go.Figure()
        for label, s_temp, s_rain, s_snow, color in scenarios:
            hours, b_vals, _ = line2_predict_day(selected_station, dt, s_temp, s_rain, s_snow, active_model, all_models, le_station, base_df)
            fig_weather.add_trace(go.Scatter(x=hours, y=b_vals, mode="lines", name=label, line=dict(color=color, width=2)))
        fig_weather.update_layout(
            title="날씨 조건별 승차인원 변화",
            xaxis=dict(title="시간", tickmode="linear", tick0=5, dtick=1, ticksuffix="시"),
            yaxis=dict(title="승차인원 (명)", tickformat=","),
            plot_bgcolor="white",
            paper_bgcolor="white",
            height=380,
            margin=dict(l=0, r=0, t=50, b=0),
        )
        fig_weather.update_yaxes(showgrid=True, gridcolor="#f1f5f9")
        st.plotly_chart(fig_weather, use_container_width=True)

    with tab3:
        st.subheader("2호선 역별 랭킹")
        with st.spinner("2호선 전체 역 예측값을 계산하는 중..."):
            rankings = []
            for station in stations:
                b, a = line2_predict(station, dt, selected_hour, temp, rain, snow, active_model, all_models, le_station, base_df)
                cong, _, _ = line2_get_congestion(b, station, avg=get_line2_avg(station, base_df))
                rankings.append({"역명": station, "승차": b, "하차": a, "혼잡도": cong})
            rank_df = pd.DataFrame(rankings).sort_values("승차", ascending=False).reset_index(drop=True)

        col_top, col_low = st.columns(2)
        col_top.dataframe(rank_df.head(10), use_container_width=True, hide_index=True)
        col_low.dataframe(rank_df.tail(10).iloc[::-1], use_container_width=True, hide_index=True)

        fig_rank = go.Figure()
        fig_rank.add_trace(go.Bar(x=rank_df.head(25)["역명"], y=rank_df.head(25)["승차"], marker_color=LINE2_COLOR))
        fig_rank.update_layout(
            title=f"상위 25개 역 승차인원 ({selected_hour}시)",
            xaxis=dict(title="역명", tickangle=-45),
            yaxis=dict(title="승차인원 (명)", tickformat=","),
            plot_bgcolor="white",
            paper_bgcolor="white",
            height=380,
            margin=dict(l=0, r=0, t=50, b=80),
        )
        st.plotly_chart(fig_rank, use_container_width=True)

    with tab4:
        st.subheader("최적 탑승 시간 추천")
        col_from, col_to = st.columns(2)
        with col_from:
            from_station = st.selectbox("출발역", stations, index=stations.index(selected_station), key="line2_from_station")
        with col_to:
            default_to = "홍대입구" if "홍대입구" in stations else stations[0]
            to_station = st.selectbox("도착역", stations, index=stations.index(default_to), key="line2_to_station")

        if st.button("2호선 최적 시간 찾기", type="primary"):
            results = []
            for hour in range(5, 24):
                b_from, _ = line2_predict(from_station, dt, hour, temp, rain, snow, active_model, all_models, le_station, base_df)
                b_to, _ = line2_predict(to_station, dt, hour, temp, rain, snow, active_model, all_models, le_station, base_df)
                results.append({"시간": hour, "출발역 승차": b_from, "도착역 승차": b_to, "합산": b_from + b_to})
            result_df = pd.DataFrame(results)
            best = result_df.loc[result_df["합산"].idxmin()]
            st.success(f"추천 탑승 시간: {int(best['시간'])}시 ({int(best['합산']):,}명 기준)")
            fig_opt = go.Figure()
            fig_opt.add_trace(go.Bar(x=result_df["시간"], y=result_df["출발역 승차"], name=from_station, marker_color="#3b82f6"))
            fig_opt.add_trace(go.Bar(x=result_df["시간"], y=result_df["도착역 승차"], name=to_station, marker_color="#10b981"))
            fig_opt.add_vline(x=best["시간"], line_dash="dot", line_color="#6366f1", annotation_text="추천")
            fig_opt.update_layout(
                title="시간대별 출발·도착역 혼잡도 비교",
                barmode="group",
                xaxis=dict(title="시간", tickmode="linear", tick0=5, dtick=1, ticksuffix="시"),
                yaxis=dict(title="승차인원 (명)", tickformat=","),
                plot_bgcolor="white",
                paper_bgcolor="white",
                height=380,
            )
            st.plotly_chart(fig_opt, use_container_width=True)

    with tab5:
        st.subheader("모델별 예측치 비교")
        comp_rows = []
        for model_name in loaded_models:
            b, a = line2_predict(selected_station, dt, selected_hour, temp, rain, snow, model_name, all_models, le_station, base_df)
            cong, _, _ = line2_get_congestion(b, selected_station, avg=get_line2_avg(selected_station, base_df))
            comp_rows.append({"모델": model_name, "예상 승차 (명)": f"{b:,}", "예상 하차 (명)": f"{a:,}", "혼잡도": cong})
        st.dataframe(pd.DataFrame(comp_rows), use_container_width=True, hide_index=True)

        col_board, col_alight = st.columns(2)
        colors = {"XGBoost": "#3b82f6", "LightGBM": "#10b981", "RandomForest": "#8b5cf6", "LSTM": "#f43f5e"}
        with col_board:
            fig_board = go.Figure()
            for model_name in loaded_models:
                hrs, b_vals, _ = line2_predict_day(selected_station, dt, temp, rain, snow, model_name, all_models, le_station, base_df)
                fig_board.add_trace(go.Scatter(x=hrs, y=b_vals, mode="lines+markers", name=model_name, line=dict(color=colors.get(model_name, "#64748b"), width=2.5)))
            fig_board.update_layout(title="모델별 24시간 승차 예측", height=380, plot_bgcolor="white", paper_bgcolor="white")
            st.plotly_chart(fig_board, use_container_width=True)
        with col_alight:
            fig_alight = go.Figure()
            for model_name in loaded_models:
                hrs, _, a_vals = line2_predict_day(selected_station, dt, temp, rain, snow, model_name, all_models, le_station, base_df)
                fig_alight.add_trace(go.Scatter(x=hrs, y=a_vals, mode="lines+markers", name=model_name, line=dict(color=colors.get(model_name, "#64748b"), width=2.5)))
            fig_alight.update_layout(title="모델별 24시간 하차 예측", height=380, plot_bgcolor="white", paper_bgcolor="white")
            st.plotly_chart(fig_alight, use_container_width=True)

        st.divider()
        st.subheader("정량 성능 비교")
        if "line2_metric_result" not in st.session_state:
            st.session_state["line2_metric_result"] = None

        if st.button("2호선 MAE 상태 계산", key="line2_metric_button"):
            metric_seed = int(np.random.randint(1, 1_000_000))
            sample = load_line2_validation_sample(seed=metric_seed, sample_size=100)
            metric_rows = []
            for model_name in loaded_models:
                try:
                    board_true, board_pred = [], []
                    alight_true, alight_pred = [], []
                    for _, row in sample.iterrows():
                        sample_dt = pd.to_datetime(row["날짜"]).to_pydatetime()
                        b, a = line2_predict(
                            row["역명"],
                            sample_dt,
                            int(row["시간"]),
                            float(row["기온"]),
                            float(row["강수량"]),
                            float(row["적설"]),
                            model_name,
                            all_models,
                            le_station,
                            base_df,
                        )
                        board_true.append(float(row["승차인원"]))
                        board_pred.append(float(b))
                        alight_true.append(float(row["하차인원"]))
                        alight_pred.append(float(a))
                    board_mae = calculate_mae(board_true, board_pred)
                    alight_mae = calculate_mae(alight_true, alight_pred)
                    avg_mae = (board_mae + alight_mae) / 2
                    metric_rows.append({
                        "모델": model_name,
                        "평균 MAE": avg_mae,
                        "상태": label_line2_validation_status(avg_mae),
                    })
                except Exception:
                    metric_rows.append({"모델": model_name, "평균 MAE": np.nan, "상태": "계산 실패"})

            metric_df = pd.DataFrame(metric_rows)[["모델", "평균 MAE", "상태"]]
            metric_df["평균 MAE"] = metric_df["평균 MAE"].map(lambda value: "-" if pd.isna(value) else f"{value:,.1f}명")
            st.session_state["line2_metric_result"] = {
                "caption": f"2호선 데이터 100개 랜덤 샘플 · 승하차 MAE 평균 · seed {metric_seed}",
                "data": metric_df,
            }

        if st.session_state["line2_metric_result"]:
            st.caption(st.session_state["line2_metric_result"]["caption"])
            st.dataframe(st.session_state["line2_metric_result"]["data"], use_container_width=True, hide_index=True)
        else:
            st.info("2호선 정량 성능 비교는 아직 계산되지 않았습니다. 버튼을 누르면 결과가 저장됩니다.")

from datetime import timedelta

import holidays as hd
import numpy as np
import pandas as pd

from core.config import FEATURE_COLS, STATION_AVG


def get_holiday_type(d, kr_hols):
    if d not in kr_hols:
        return 0

    name = kr_hols.get(d, "")
    if any(k in name for k in ["Korean New Year", "Chuseok", "설날", "추석", "preceding", "second day"]):
        return 3
    if any(k in name for k in ["Children", "Christmas", "Buddha", "어린이날", "성탄절", "기독탄신일", "부처님오신날", "석가탄신일"]):
        return 2
    return 1


def _station_key_parts(station):
    if "_" in station:
        base, line = station.rsplit("_", 1)
        return base, line
    return station, ""


def _safe_label_transform(encoder, value):
    if encoder is None or not hasattr(encoder, "classes_"):
        return 0
    if value not in encoder.classes_:
        return 0
    return int(encoder.transform([value])[0])


def get_station_avg(station, lstm_base_df=None, fallback=1500):
    if lstm_base_df is not None and not lstm_base_df.empty and "역명" in lstm_base_df.columns:
        matched = lstm_base_df[lstm_base_df["역명"] == station]
        if not matched.empty and "승차인원" in matched.columns:
            return float(np.expm1(matched["승차인원"]).mean())

    return float(STATION_AVG.get(station, fallback))


def _station_log_averages(station, lstm_base_df=None, fallback=1500):
    if lstm_base_df is not None and not lstm_base_df.empty and "역명" in lstm_base_df.columns:
        matched = lstm_base_df[lstm_base_df["역명"] == station]
        if not matched.empty:
            board_avg = float(matched["역별_평균_승차"].mean())
            alight_avg = float(matched["역별_평균_하차"].mean())
            return board_avg, alight_avg

    fallback_log = float(np.log1p(STATION_AVG.get(station, fallback)))
    return fallback_log, fallback_log


def predict(station, dt, hour, temp, rain=0.0, snow=0.0, model_name="LSTM", all_models=None, le_station=None, lstm_base_df=None):
    if all_models is None:
        raise ValueError("models must be provided.")

    m_info = all_models.get(model_name, {})
    if not m_info.get("loaded", False):
        raise ValueError(f"Model '{model_name}' is not loaded. Please verify model files in models/ directory.")

    kr_hols = hd.KR(years=dt.year)
    is_hol = int(dt.date() in kr_hols)
    stn_enc = le_station.transform([station])[0] if le_station is not None and station in le_station.classes_ else 0

    prev_d = dt - timedelta(days=1)
    next_d = dt + timedelta(days=1)
    prev_off = (prev_d.weekday() >= 5) or (prev_d.date() in kr_hols)
    next_off = (next_d.weekday() >= 5) or (next_d.date() in kr_hols)
    is_off = (dt.weekday() >= 5) or bool(is_hol)

    row = {
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
        "비근무일": int(dt.weekday() >= 5 or bool(is_hol)),
        "출근피크": int(hour in [7, 8, 9]),
        "퇴근피크": int(hour in [18, 19, 20]),
        "기온": temp,
        "강수량": rain,
        "적설": snow,
        "강수_여부": int(rain > 0),
        "적설_여부": int(snow > 0),
        "불쾌지수": round(9 / 5 * temp - 0.55 * (1 - int(rain > 0) * 0.8) * (9 / 5 * temp - 26) + 32, 2),
        "공휴일_유형": get_holiday_type(dt.date(), kr_hols),
        "연휴_여부": int(is_off and (prev_off or next_off)),
        "공휴일_전날": int(next_d.date() in kr_hols),
        "공휴일_다음날": int(prev_d.date() in kr_hols),
    }

    X = pd.DataFrame([row])[FEATURE_COLS]

    if model_name == "LSTM" and m_info.get("loaded", False):
        try:
            scaler = m_info["scaler"]
            le_lstm = m_info.get("le")
            le_line = m_info.get("le_line")

            station_idx = 0
            if le_lstm is not None:
                station_idx = int(le_lstm.transform([station])[0])

            line_name = _station_key_parts(station)[1] or "2호선"
            line_idx = 0
            if le_line is not None:
                try:
                    line_idx = int(le_line.transform([line_name])[0])
                except Exception:
                    line_idx = 0

            log_avg_board, log_avg_alight = _station_log_averages(station, lstm_base_df)

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
                "역별_평균_승차": float(log_avg_board),
                "역별_평균_하차": float(log_avg_alight),
            }

            scaler_cols = [
                "승차인원",
                "하차인원",
                "시간",
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
            df_scale_in = pd.DataFrame([row_scaler])[scaler_cols]
            df_scaled = pd.DataFrame(scaler.transform(df_scale_in), columns=scaler_cols)
            feat_12 = list(df_scaled.values[0])

            if lstm_base_df is not None:
                station_all = lstm_base_df[lstm_base_df["역명"] == station].copy().reset_index(drop=True)
                same_hour_idx = station_all[station_all["시간"] == hour].index.tolist()

                if len(same_hour_idx) == 0 and not station_all.empty:
                    same_hour_idx = [int(np.abs(station_all["시간"] - hour).idxmin())]

                if station_all.empty:
                    seq = np.array([feat_12] * 12)
                else:
                    best_idx = same_hour_idx[-1] if same_hour_idx else min(len(station_all) - 1, 11)
                    start_idx = max(0, best_idx - 12)
                    seq_data = station_all.iloc[start_idx:best_idx]
                    if len(seq_data) < 12:
                        seq_data = station_all.iloc[best_idx:best_idx + 12]

                    seq_raw = seq_data.tail(12)[scaler_cols].values.copy()
                    seq = scaler.transform(seq_raw)
                    if len(seq) == 12:
                        seq[-1] = feat_12
                    elif len(seq) < 12:
                        pad = np.array([feat_12] * (12 - len(seq)))
                        seq = np.vstack([pad, seq])[-12:]
            else:
                seq = np.array([feat_12] * 12)

            X_in = np.expand_dims(seq, axis=0)
            stn_in = np.array([[float(station_idx)]])
            line_in = np.array([[float(line_idx)]])

            pred_b = np.array(m_info["board"]([X_in, stn_in, line_in], training=False))

            dummy_row_b = np.zeros((1, 12))
            dummy_row_b[0, 0] = pred_b.flatten()[0]
            log_val_b = scaler.inverse_transform(dummy_row_b)[0, 0]

            boarding_pred_raw = float(pred_b.flatten()[0])

            pred_a_inputs = [X_in, stn_in, line_in, np.array([[boarding_pred_raw]], dtype=np.float32)]
            pred_a = np.array(m_info["alight"](pred_a_inputs, training=False))

            dummy_row_a = np.zeros((1, 12))
            dummy_row_a[0, 1] = pred_a.flatten()[0]
            log_val_a = scaler.inverse_transform(dummy_row_a)[0, 1]

            b_val = np.expm1(log_val_b)
            a_val = np.expm1(log_val_a)

            b = int(np.clip(b_val, 0, None))
            a = int(np.clip(a_val, 0, None))
        except Exception as e:
            raise RuntimeError(f"LSTM prediction failed: {e}")
    elif model_name == "LightGBM":
        station_name, line_name = _station_key_parts(station)
        columns = m_info.get("columns") or ["역명", "호선", "시간", "요일", "월", "공휴일여부", "기온", "강수량", "적설", "year", "day", "weekday"]
        row_lgb = {
            "역명": station_name,
            "호선": line_name,
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
        }
        X_lgb = pd.DataFrame([row_lgb])[columns]

        categories = getattr(m_info["board"].booster_, "pandas_categorical", None)
        if categories and len(categories) >= 2:
            X_lgb["역명"] = pd.Categorical(X_lgb["역명"], categories=categories[0])
            X_lgb["호선"] = pd.Categorical(X_lgb["호선"], categories=categories[1])

        b = int(np.clip(m_info["board"].predict(X_lgb), 0, None)[0])
        a = int(np.clip(m_info["alight"].predict(X_lgb), 0, None)[0])
    elif model_name == "XGBoost":
        station_name, line_name = _station_key_parts(station)
        xgb_le_station = m_info.get("le_station")
        xgb_le_line = m_info.get("le_line")
        columns = m_info.get("columns") or list(getattr(m_info["board"], "feature_names_in_", []))
        if not columns:
            columns = [
                "역명_enc", "호선_enc", "시간", "시간_sin", "시간_cos",
                "요일", "요일_sin", "요일_cos", "월", "월_sin", "월_cos",
                "공휴일여부", "비근무일", "출근피크", "퇴근피크",
                "기온", "강수량", "적설", "강수_여부", "적설_여부",
                "불쾌지수", "공휴일_유형", "연휴_여부", "공휴일_전날", "공휴일_다음날",
            ]

        row_xgb = {
            **row,
            "역명_enc": _safe_label_transform(xgb_le_station, station_name),
            "호선_enc": _safe_label_transform(xgb_le_line, line_name),
        }
        X_xgb = pd.DataFrame([row_xgb])[columns]

        b = int(np.clip(m_info["board"].predict(X_xgb), 0, None)[0])
        a = int(np.clip(m_info["alight"].predict(X_xgb), 0, None)[0])
    else:
        raise ValueError(f"Model '{model_name}' is not supported in the 1-8 line app.")

    return b, a


def predict_day(station, dt, temp, rain=0.0, snow=0.0, model_name="LSTM", all_models=None, le_station=None, lstm_base_df=None):
    hours, boards, alights = [], [], []
    for hr in range(5, 24):
        b, a = predict(station, dt, hr, temp, rain, snow, model_name, all_models, le_station, lstm_base_df)
        hours.append(hr)
        boards.append(b)
        alights.append(a)
    return hours, boards, alights


def get_congestion(board, station, avg=None):
    if avg is None:
        avg = STATION_AVG.get(station, 1500)

    ratio = board / avg if avg > 0 else 1
    if ratio < 0.6:
        return "쾌적", "#065f46", "#d1fae5"
    if ratio < 1.0:
        return "보통", "#1e40af", "#dbeafe"
    if ratio < 1.5:
        return "혼잡", "#92400e", "#fef3c7"
    return "매우혼잡", "#991b1b", "#fee2e2"

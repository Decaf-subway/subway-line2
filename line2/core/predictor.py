import pandas as pd
import numpy as np
import holidays as hd
from datetime import timedelta
from core.config import FEATURE_COLS, STATION_AVG

def get_holiday_type(d, kr_hols):
    if d not in kr_hols:
        return 0
    name = kr_hols.get(d, "")
    if any(k in name for k in ["Korean New Year", "Chuseok", "설날", "추석", "preceding", "second day"]):
        return 3
    elif any(k in name for k in ["Children", "Christmas", "Buddha", "어린이날", "성탄절", "기독탄신일", "부처님오신날", "석가탄신일"]):
        return 2
    return 1

def predict(station, dt, hour, temp, rain=0.0, snow=0.0, model_name="XGBoost", all_models=None, le_station=None, lstm_base_df=None):
    if all_models is None:
        raise ValueError("models must be provided.")
        
    m_info = all_models.get(model_name, {})
    if not m_info.get("loaded", False):
        raise ValueError(f"Model '{model_name}' is not loaded. Please verify model files in models/ directory.")

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
    
    if model_name == "LSTM" and m_info.get("loaded", False):
        try:
            scaler = m_info["scaler"]
            is_jamsil = "잠실" in station
            if is_jamsil and "board_jamsil" in m_info and "alight_jamsil" in m_info:
                m_board = m_info["board_jamsil"]
                m_alight = m_info["alight_jamsil"]
            else:
                m_board = m_info["board"]
                m_alight = m_info["alight"]

            station_idx = 0
            le_lstm = m_info.get("le")
            if le_lstm is not None:
                for idx, cls in enumerate(le_lstm.classes_):
                    if station in cls or cls in station:
                        station_idx = idx
                        break

            avg_val = 1500
            if le_lstm is not None:
                matched_cls = le_lstm.classes_[station_idx]
                for k, v in STATION_AVG.items():
                    if k in matched_cls or matched_cls in k:
                        avg_val = v
                        break
                
            log_avg_val = np.log1p(avg_val)

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
            
            scaler_cols = ['승차인원', '하차인원', '시간', '요일', '월', '공휴일여부', '기온', '강수량', '적설', '요일_시간', '역별_평균_승차', '역별_평균_하차']
            df_scale_in = pd.DataFrame([row_scaler])[scaler_cols]
            df_scaled = pd.DataFrame(scaler.transform(df_scale_in), columns=scaler_cols)
            
            feat_12 = list(df_scaled.values[0])
            
            if lstm_base_df is not None:
                matched_stn = station
                le_lstm = m_info.get("le")
                if le_lstm is not None:
                    matched_stn = le_lstm.classes_[station_idx]
                
                station_all = lstm_base_df[lstm_base_df['역명'] == matched_stn].copy().reset_index(drop=True)
                
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
            stn_in = np.array([[float(station_idx)]])
            
            pred_b = np.array(m_board([X_in, stn_in], training=False))
            pred_a = np.array(m_alight([X_in, stn_in], training=False))
            
            dummy_row_b = np.zeros((1, 12))
            dummy_row_b[0, 0] = pred_b.flatten()[0]
            log_val_b = scaler.inverse_transform(dummy_row_b)[0, 0]
            
            dummy_row_a = np.zeros((1, 12))
            dummy_row_a[0, 1] = pred_a.flatten()[0]
            log_val_a = scaler.inverse_transform(dummy_row_a)[0, 1]
            
            b_val = np.expm1(log_val_b)
            a_val = np.expm1(log_val_a)
            
            b = int(np.clip(b_val, 0, None))
            a = int(np.clip(a_val, 0, None))
            
        except Exception as e:
            xgb_info = all_models.get("XGBoost", {})
            if xgb_info.get("loaded", False):
                b = int(np.clip(xgb_info["board"].predict(X), 0, None)[0])
                a = int(np.clip(xgb_info["alight"].predict(X), 0, None)[0])
            else:
                raise RuntimeError(f"LSTM prediction failed: {e}, and XGBoost fallback is unavailable.")
    else:
        if model_name == "RandomForest":
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
            for col in rf_cols:
                if col.startswith("역명_"):
                    stn_part = col.replace("역명_", "")
                    row_rf[col] = 1.0 if stn_part == station else 0.0
            
            X_rf = pd.DataFrame([row_rf])[rf_cols]
            b = int(np.clip(m_info["board"].predict(X_rf),  0, None)[0])
            a = int(np.clip(m_info["alight"].predict(X_rf), 0, None)[0])
        elif model_name == "LightGBM":
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
            
            stn_categories = sorted(list(le_station.classes_))
            line_categories = ['2호선']
            df_lgb["역명"] = pd.Categorical(df_lgb["역명"], categories=stn_categories)
            df_lgb["호선"] = pd.Categorical(df_lgb["호선"], categories=line_categories)
            
            lgb_cols = ['역명', '호선', '시간', '요일', '월', '공휴일여부', '기온', '강수량', '적설', 'year', 'day', 'weekday']
            X_lgb = df_lgb[lgb_cols]
            
            b = int(np.clip(m_info["board"].predict(X_lgb),  0, None)[0])
            a = int(np.clip(m_info["alight"].predict(X_lgb), 0, None)[0])
        else:
            b = int(np.clip(m_info["board"].predict(X),  0, None)[0])
            a = int(np.clip(m_info["alight"].predict(X), 0, None)[0])
        
    return b, a

def predict_day(station, dt, temp, rain=0.0, snow=0.0, model_name="XGBoost", all_models=None, le_station=None, lstm_base_df=None):
    hours, boards, alights = [], [], []
    for hr in range(5, 24):
        b, a = predict(station, dt, hr, temp, rain, snow, model_name, all_models, le_station, lstm_base_df)
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

import requests
import holidays as hd
from datetime import timedelta
from core.predictor import get_holiday_type

def get_realtime_train_positions(api_key="sample"):
    try:
        url = f"http://swopenAPI.seoul.go.kr/api/subway/{api_key}/json/realtimePosition/0/50/2호선"
        res = requests.get(url, timeout=5)
        if res.status_code == 200:
            data = res.json()
            if "realtimePositionList" in data:
                return data["realtimePositionList"]
    except Exception:
        pass
    return []

def get_realtime_station_arrival(station_name, api_key="sample"):
    try:
        pure_name = station_name.split("(")[0]
        url = f"http://swopenAPI.seoul.go.kr/api/subway/{api_key}/json/realtimeStationArrival/0/20/{pure_name}"
        res = requests.get(url, timeout=5)
        if res.status_code == 200:
            data = res.json()
            if "realtimeArrivalList" in data:
                return [arr for arr in data["realtimeArrivalList"] if arr.get("subwayId") == "1002"]
    except Exception:
        pass
    return []

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

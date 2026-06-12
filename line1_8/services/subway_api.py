from datetime import timedelta
import re
from urllib.parse import quote

import holidays as hd
import requests

LINE_API_NAMES = {
    "1호선": "1호선",
    "2호선": "2호선",
    "3호선": "3호선",
    "4호선": "4호선",
    "5호선": "5호선",
    "6호선": "6호선",
    "7호선": "7호선",
    "8호선": "8호선",
    "1?몄꽑": "1호선",
    "2?몄꽑": "2호선",
    "3?몄꽑": "3호선",
    "4?몄꽑": "4호선",
    "5?몄꽑": "5호선",
    "6?몄꽑": "6호선",
    "7?몄꽑": "7호선",
    "8?몄꽑": "8호선",
}


def _line_name_to_api_name(line_name):
    if line_name in LINE_API_NAMES:
        return LINE_API_NAMES[line_name]

    digits = "".join(ch for ch in str(line_name) if ch.isdigit())
    return f"{digits}호선" if digits else "2호선"


def get_realtime_train_positions(api_key="sample", line_name="2호선"):
    try:
        subway_name = quote(_line_name_to_api_name(line_name))
        url = f"http://swopenAPI.seoul.go.kr/api/subway/{api_key}/json/realtimePosition/0/50/{subway_name}"
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
        pure_name = quote(station_name.split("(")[0].strip())
        url = f"http://swopenAPI.seoul.go.kr/api/subway/{api_key}/json/realtimeStationArrival/0/20/{pure_name}"
        res = requests.get(url, timeout=5)
        if res.status_code == 200:
            data = res.json()
            if "realtimeArrivalList" in data:
                cleaned = []
                for arr in data["realtimeArrivalList"]:
                    item = dict(arr)
                    msg = str(item.get("arvlMsg2", "")).strip()
                    msg = re.sub(r"\s*\d+번째\s*전역.*$", "", msg)
                    msg = re.sub(r"\s*\d+번째전역.*$", "", msg)
                    item["arvlMsg2"] = msg.strip()
                    cleaned.append(item)
                return cleaned
    except Exception:
        pass
    return []


def check_holiday(dt):
    kr_hols = hd.KR(years=dt.year)
    if dt.date() in kr_hols:
        name = kr_hols.get(dt.date(), "공휴일")
        return name, f"오늘은 {name}입니다."

    prev_d = dt - timedelta(days=1)
    next_d = dt + timedelta(days=1)
    if next_d.date() in kr_hols:
        return None, f"내일은 공휴일({kr_hols.get(next_d.date())})입니다."
    if prev_d.date() in kr_hols:
        return None, f"어제는 공휴일({kr_hols.get(prev_d.date())})이었습니다."
    return None, None

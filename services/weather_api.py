import requests
from datetime import datetime, date, timedelta

def get_kma_weather(api_key, dt, hour):
    """
    기상청 공공데이터 포털 단기예보 API (서울 기준 nx=60, ny=127).
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

        temp = float(result.get("TMP", 15.0))
        rain = parse_kma_value(result.get("PCP", "0"))
        snow = parse_kma_value(result.get("SNO", "0"))

        return round(temp, 1), round(rain, 1), round(snow, 1), "단기예보"

    except Exception as e:
        print(f"[경고] 기상청 단기 예보 호출 실패: {e}")
        return None

def get_mid_forecast_time():
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
    기상청 공공데이터 포털 중기예보 API (서울 육상코드 11B00000, 기온코드 11B10101).
    """
    try:
        target_date = dt if isinstance(dt, date) else dt.date()
        days_diff = (target_date - date.today()).days
        if days_diff < 3 or days_diff > 10:
            return None

        tm_fc = get_mid_forecast_time()

        url_ta = (
            "http://apis.data.go.kr/1360000/MidFcstInfoService/getMidTa"
            f"?serviceKey={api_key}"
            f"&pageNo=1&numOfRows=10&dataType=JSON"
            f"&regId=11B10101"
            f"&tmFc={tm_fc}"
        )

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

        ta_min = float(item_ta.get(f"taMin{days_diff}", 10.0))
        ta_max = float(item_ta.get(f"taMax{days_diff}", 20.0))
        avg_temp = (ta_min + ta_max) / 2.0

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

    except Exception:
        return None

def get_weather_with_fallback(api_key, dt, hour):
    target_date = dt if isinstance(dt, date) else dt.date()
    days_diff = (target_date - date.today()).days
    
    if 0 <= days_diff <= 2:
        res = get_kma_weather(api_key, dt, hour)
        if res is not None:
            return res
            
    if 3 <= days_diff <= 10:
        res_mid = get_kma_mid_weather(api_key, dt)
        if res_mid is not None:
            return res_mid
            
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

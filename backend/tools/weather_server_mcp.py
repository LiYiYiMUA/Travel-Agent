#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
MCP 鏈嶅姟鍣ㄥ彲浠ユ彁渚涗笁绉嶄富瑕佺被鍨嬬殑鍔熻兘锛?
璧勬簮锛氬鎴风鍙互璇诲彇鐨勭被浼兼枃浠剁殑鏁版嵁锛堜緥濡?API 鍝嶅簲鎴栨枃浠跺唴瀹癸級
宸ュ叿锛氬彲鐢?LLM 璋冪敤鐨勫嚱鏁帮紙缁忕敤鎴锋壒鍑嗭級
鎻愮ず锛氶鍏堢紪鍐欑殑妯℃澘锛屽府鍔╃敤鎴峰畬鎴愮壒瀹氫换鍔?
######################################

MCP 澶╂皵鏈嶅姟鍣?
鎻愪緵涓や釜宸ュ叿锛?1. get_weather_warning: 鑾峰彇鎸囧畾鍩庡競ID鎴栫粡绾害鐨勫ぉ姘旂伨瀹抽璀?2. get_daily_forecast: 鑾峰彇鎸囧畾鍩庡競ID鎴栫粡绾害鐨勫ぉ姘旈鎶?
Author: FlyAIBox
Date: 2025.10.11
"""

from typing import Any, Dict, List, Optional, Union
import logging
from pathlib import Path
import asyncio
import httpx
import os
import re
from urllib.parse import urljoin
from mcp.server.fastmcp import FastMCP
from dotenv import load_dotenv
from pathlib import Path
from pypinyin import lazy_pinyin, Style

# 加载 .env 文件中的环境变量
dotenv_path = Path(__file__).resolve().parents[1] / '.env'
load_dotenv(dotenv_path, override=True)

# 初始化日志
def setup_weather_server_logger():
    ws_logger = logging.getLogger('weather_server')
    ws_logger.setLevel(logging.INFO)
    ws_logger.propagate = False
    if not ws_logger.handlers:
        log_dir = Path(__file__).resolve().parents[1] / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_dir / "backend.log", encoding='utf-8')
        fh.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                                      datefmt='%Y-%m-%d %H:%M:%S')
        fh.setFormatter(formatter)
        ws_logger.addHandler(fh)
    return ws_logger

ws_logger = setup_weather_server_logger()

# 初始化 FastMCP 服务
mcp = FastMCP(
    "weather",
    debug=False,
    host="0.0.0.0",
)

# 浠庣幆澧冨彉閲忎腑璇诲彇甯搁噺
QWEATHER_API_BASE = os.getenv("QWEATHER_API_BASE")
QWEATHER_API_KEY = os.getenv("QWEATHER_API_KEY")

def _normalize_base_url(raw_base: Optional[str]) -> str:
    """
    纭繚鍩虹 URL 鍖呭惈鍗忚骞朵互鍗曚釜鏂滄潬缁撳熬锛屽吋瀹?.env 涓湭鍐欏崗璁殑鎯呭喌
    """
    if not raw_base:
        raise RuntimeError("鏈厤缃?QWEATHER_API_BASE 鐜鍙橀噺")

    base = raw_base.strip()
    if not base.startswith(("http://", "https://")):
        base = f"https://{base.lstrip('/')}"

    # urljoin 要求目录风格以斜杠结尾，避免 'v7/weather/7d' 被覆盖
    if not base.endswith("/"):
        base = f"{base}/"

    return base

try:
    _QWEATHER_BASE_URL = _normalize_base_url(QWEATHER_API_BASE)
except RuntimeError as err:
    print(f"[閰嶇疆閿欒] {err}")
    _QWEATHER_BASE_URL = None

async def make_qweather_request(endpoint: str, params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    鍚戝拰椋庡ぉ姘?API 鍙戦€佽姹?    
    鍙傛暟:
        endpoint: API 绔偣璺緞锛堜笉鍖呭惈鍩虹 URL锛?        params: API 璇锋眰鐨勫弬鏁?        
    杩斿洖:
        鎴愬姛鏃惰繑鍥?JSON 鍝嶅簲锛屽け璐ユ椂杩斿洖 None
    """
    if not _QWEATHER_BASE_URL:
        ws_logger.error("QWEATHER_API_BASE 未正确配置，已跳过请求。")
        return None

    if not QWEATHER_API_KEY:
        ws_logger.error("QWEATHER_API_KEY 未设置，已跳过请求。")
        return None

    safe_endpoint = endpoint.lstrip("/")
    url = urljoin(_QWEATHER_BASE_URL, safe_endpoint)

    # 使用 Header 方式认证（和风天气新版 API）
    headers = {
        "X-QW-Api-Key": QWEATHER_API_KEY
    }
    
    async with httpx.AsyncClient() as client:
        try:
            ws_logger.info(f"QWeather璇锋眰: url={url}, params={params}")
            response = await client.get(url, params=params, headers=headers, timeout=30.0)
            ws_logger.info(f"QWeather鍝嶅簲鐘舵€? {response.status_code}")
            response.raise_for_status()
            result = response.json()
            ws_logger.info(f"QWeather鍝嶅簲鍐呭澶у皬: {len(str(result))} 瀛楃")
            return result
        except httpx.HTTPStatusError as e:
            ws_logger.error(f"HTTP 鐘舵€侀敊璇? {e.response.status_code} - {e.response.text}")
            return None
        except Exception as e:
            ws_logger.error(f"API 璇锋眰閿欒: {type(e).__name__}: {e}")
            return None

def format_warning(warning: Dict[str, Any]) -> str:
    """
    灏嗗ぉ姘旈璀︽暟鎹牸寮忓寲涓哄彲璇诲瓧绗︿覆
    
    鍙傛暟:
        warning: 澶╂皵棰勮鏁版嵁瀵硅薄
        
    杩斿洖:
        鏍煎紡鍖栧悗鐨勯璀︿俊鎭?    """
    return f"""
棰勮ID: {warning.get('id', '鏈煡')}
鏍囬: {warning.get('title', '鏈煡')}
鍙戝竷鏃堕棿: {warning.get('pubTime', '鏈煡')}
寮€濮嬫椂闂? {warning.get('startTime', '鏈煡')}
缁撴潫鏃堕棿: {warning.get('endTime', '鏈煡')}
棰勮绫诲瀷: {warning.get('typeName', '鏈煡')}
棰勮绛夌骇: {warning.get('severity', '鏈煡')} ({warning.get('severityColor', '鏈煡')})
鍙戝竷鍗曚綅: {warning.get('sender', '鏈煡')}
鐘舵€? {warning.get('status', '鏈煡')}
详情信息: {warning.get('text', '无详细信息')}
"""

def _contains_chinese(text: str) -> bool:
    """Return True if text contains any CJK unified ideograph."""
    return any('\u4e00' <= ch <= '\u9fff' for ch in text)


def _english_lookup_candidates(text: str) -> List[str]:
    """
    Build fallback candidates for city lookup from English inputs.
    Example: "New York" -> ["new york", "newyork"].
    """
    normalized = text.strip().lower()
    candidates: List[str] = []
    if normalized:
        candidates.append(normalized)

    compact = re.sub(r"[^a-z0-9,]", "", normalized)
    if compact and compact not in candidates:
        candidates.append(compact)
    return candidates


async def _resolve_qweather_location(raw: Union[str, int], label: str) -> str:
    """
    Resolve user input into a QWeather-acceptable location:
    - lat,lon -> pass-through
    - numeric city id -> pass-through
    - Chinese/English city name -> geo lookup -> city id
    """
    text = str(raw).strip()

    if "," in text:
        ws_logger.info(f"[{label}]妫€娴嬪埌缁忕含搴︼紝鐩存帴浣跨敤: {text}")
        return text

    if text.isdigit():
        ws_logger.info(f"[{label}]妫€娴嬪埌鍩庡競ID锛岀洿鎺ヤ娇鐢? {text}")
        return text

    candidates: List[str] = []
    if _contains_chinese(text):
        py = _convert_chinese_to_pinyin(text)
        ws_logger.info(f"[{label}]涓枃杞嫾闊? {text} -> {py}")
        candidates.append(py)
    elif any(ch.isalpha() for ch in text):
        candidates.extend(_english_lookup_candidates(text))
    else:
        ws_logger.info(f"[{label}]鏈瘑鍒殑鏍煎紡锛屽師鏍蜂娇鐢? {text}")
        return text

    for candidate in candidates:
        ws_logger.info(f"[{label}]灏濊瘯鍩庡競妫€绱? {candidate}")
        lookup = await make_qweather_request(
            "geo/v2/city/lookup",
            {"location": candidate, "lang": "zh"}
        )
        if not lookup or lookup.get("code") != "200":
            ws_logger.warning(f"[{label}]鍩庡競妫€绱㈠け璐? {candidate}")
            continue

        locations = lookup.get("location", [])
        if not locations:
            ws_logger.info(f"[{label}]鍩庡競妫€绱㈡棤缁撴灉: {candidate}")
            continue

        chosen = next((loc for loc in locations if loc.get("type") == "city"), locations[0])
        city_id = chosen.get("id")
        if city_id:
            ws_logger.info(f"[{label}]瑙ｆ瀽瀹屾垚: {text} -> {city_id}")
            return city_id

    ws_logger.warning(f"[{label}]鏃犳硶瑙ｆ瀽鍩庡競ID锛屽洖閫€鍘熷€? {text}")
    return text


@mcp.tool()
async def get_weather_warning(location: Union[str, int]) -> str:
    """
    获取指定位置的天气灾害预警。
    """
    resolved = await _resolve_qweather_location(location, "预警")

    params = {
        "location": resolved,
        "lang": "zh"
    }

    ws_logger.info(f"调用 get_weather_warning | params={params}")
    data = await make_qweather_request("v7/warning/now", params)

    if not data:
        ws_logger.warning("get_weather_warning 返回空或失败")
        return "无法获取预警信息或API请求失败。"

    if data.get("code") != "200":
        ws_logger.error(f"get_weather_warning API错误: {data.get('code')}")
        return f"API 返回错误: {data.get('code')}"

    warnings = data.get("warning", [])

    if not warnings:
        ws_logger.info(f"get_weather_warning 无活动预警 | location={location}")
        return f"当前位置 {location} 没有活动预警。"

    formatted_warnings = [format_warning(warning) for warning in warnings]
    joined = "\n---\n".join(formatted_warnings)
    ws_logger.info(f"get_weather_warning 返回长度: {len(joined)} 字符")
    return joined


def format_daily_forecast(daily: Dict[str, Any]) -> str:
    """
    灏嗗ぉ姘旈鎶ユ暟鎹牸寮忓寲涓哄彲璇诲瓧绗︿覆
    
    鍙傛暟:
        daily: 澶╂皵棰勬姤鏁版嵁瀵硅薄
        
    杩斿洖:
        鏍煎紡鍖栧悗鐨勯鎶ヤ俊鎭?    """
    return f"""
鏃ユ湡: {daily.get('fxDate', '鏈煡')}
鏃ュ嚭: {daily.get('sunrise', '鏈煡')}  鏃ヨ惤: {daily.get('sunset', '鏈煡')}
鏈€楂樻俯搴? {daily.get('tempMax', '鏈煡')}掳C  鏈€浣庢俯搴? {daily.get('tempMin', '鏈煡')}掳C
鐧藉ぉ澶╂皵: {daily.get('textDay', '鏈煡')}  澶滈棿澶╂皵: {daily.get('textNight', '鏈煡')}
鐧藉ぉ椋庡悜: {daily.get('windDirDay', '鏈煡')} {daily.get('windScaleDay', '鏈煡')}绾?({daily.get('windSpeedDay', '鏈煡')}km/h)
澶滈棿椋庡悜: {daily.get('windDirNight', '鏈煡')} {daily.get('windScaleNight', '鏈煡')}绾?({daily.get('windSpeedNight', '鏈煡')}km/h)
鐩稿婀垮害: {daily.get('humidity', '鏈煡')}%
闄嶆按閲? {daily.get('precip', '鏈煡')}mm
绱绾挎寚鏁? {daily.get('uvIndex', '鏈煡')}
鑳借搴? {daily.get('vis', '鏈煡')}km
"""

@mcp.tool()
async def get_daily_forecast(location: Union[str, int], days: int = 3) -> str:
    """
    获取指定位置的天气预报。
    """
    resolved = await _resolve_qweather_location(location, "预报")

    valid_days = [3, 7, 10, 15, 30]
    if days not in valid_days:
        days = 3

    params = {
        "location": resolved,
        "lang": "zh"
    }

    endpoint = f"v7/weather/{days}d"
    ws_logger.info(f"调用 get_daily_forecast | endpoint={endpoint}, params={params}")
    data = await make_qweather_request(endpoint, params)

    if not data:
        ws_logger.warning("get_daily_forecast 返回空或失败")
        return "无法获取天气预报或API请求失败。"

    if data.get("code") != "200":
        ws_logger.error(f"get_daily_forecast API错误: {data.get('code')}")
        return f"API 返回错误: {data.get('code')}"

    daily_forecasts = data.get("daily", [])

    if not daily_forecasts:
        ws_logger.warning(f"get_daily_forecast 无数据 | location={location}")
        return f"无法获取 {location} 的天气预报数据。"

    formatted_forecasts = [format_daily_forecast(daily) for daily in daily_forecasts]
    joined = "\n---\n".join(formatted_forecasts)
    ws_logger.info(f"get_daily_forecast 返回长度: {len(joined)} 字符")
    return joined


def _convert_chinese_to_pinyin(chinese_text: str) -> str:
    """
    灏嗕腑鏂囧煄甯傚悕杞崲涓烘嫾闊筹紙鍏ㄦ嫾锛?    
    Args:
        chinese_text: 涓枃鍩庡競鍚嶏紝濡?"瑗垮畞"
        
    Returns:
        str: 鎷奸煶鍏ㄦ嫾锛屽 "xining"
    """
    try:
        # 浣跨敤 pypinyin 灏嗕腑鏂囪浆鎹负鎷奸煶
        pinyin_list = lazy_pinyin(chinese_text, style=Style.NORMAL)
        pinyin = ''.join(pinyin_list)
        ws_logger.info(f"涓枃杞嫾闊? {chinese_text} 鈫?{pinyin}")
        return pinyin
    except Exception as e:
        ws_logger.error(f"涓枃杞嫾闊冲け璐? {chinese_text} - {str(e)}")
        return chinese_text  # 杞崲澶辫触鏃惰繑鍥炲師鏂囨湰


async def lookup_city_id_by_pinyin(pinyin: str) -> str:
    """
    鏍规嵁鍩庡競鍚嶇О鐨勬嫾闊筹紙鍏ㄦ嫾锛夋煡鎵惧煄甯侷D銆?
    鍙傛暟:
        pinyin: 鍩庡競鍚嶇О鐨勬嫾闊筹紙鍏ㄦ嫾锛夛紝濡?"xining"

    杩斿洖:
        鑻ユ垚鍔燂紝杩斿洖鍖归厤鍩庡競瀵硅薄鐨勭簿绠€ JSON 瀛楃涓诧紙鍖呭惈 name銆乮d銆乴at銆乴on銆乤dm1 绛夊瓧娈碉級锛?        鑻ュけ璐ユ垨鏈壘鍒帮紝杩斿洖璇存槑鏂囨湰銆?    """
    params = {
        "location": pinyin,
        "lang": "zh"
    }

    endpoint = "geo/v2/city/lookup"
    ws_logger.info(f"璋冪敤 [鏌ユ壘鍩庡競ID]| endpoint={endpoint}, params={params}")
    data = await make_qweather_request(endpoint, params)

    if not data:
        ws_logger.warning("[鏌ユ壘鍩庡競ID]杩斿洖绌烘垨澶辫触")
        return "无法查询城市ID或API请求失败。"

    if data.get("code") != "200":
        ws_logger.error(f"[鏌ユ壘鍩庡競ID]API閿欒: {data.get('code')}")
        return f"API 杩斿洖閿欒: {data.get('code')}"

    locations = data.get("location", [])
    if not locations:
        ws_logger.info(f"[鏌ユ壘鍩庡競ID]鏃犲尮閰嶇粨鏋?| pinyin={pinyin}")
        return f"未找到与 {pinyin} 匹配的城市。"

    # 浼樺厛閫夋嫨 type == "city" 鐨勪富鍩庡競锛屽惁鍒欏洖閫€绗竴涓?    chosen = None
    for loc in locations:
        if loc.get("type") == "city":
            chosen = loc
            break
    if chosen is None:
        chosen = locations[0]

    # 浠呰繑鍥炲父鐢ㄥ瓧娈碉紝閬垮厤鍐椾綑
    result = {
        "name": chosen.get("name"),
        "id": chosen.get("id"),
        "lat": chosen.get("lat"),
        "lon": chosen.get("lon"),
        "adm2": chosen.get("adm2"),
        "adm1": chosen.get("adm1"),
        "country": chosen.get("country"),
        "type": chosen.get("type"),
        "rank": chosen.get("rank"),
        "fxLink": chosen.get("fxLink"),
    }

    ws_logger.info(
        f"[鏌ユ壘鍩庡競ID]鍛戒腑: name={result['name']}, id={result['id']}"
    )

    # 以紧凑 JSON 字符串形式返回
    try:
        import json as _json
        return _json.dumps(result, ensure_ascii=False)
    except Exception:
        # 鍏滃簳涓哄彲璇诲瓧绗︿覆
        return f"{result}"

if __name__ == "__main__":
    ws_logger.info("正在启动 MCP 天气服务器...")
    ws_logger.info("鎻愪緵宸ュ叿: get_weather_warning, get_daily_forecast")
    ws_logger.info("请确保环境变量 QWEATHER_API_KEY 已设置")
    ws_logger.info("使用 Ctrl+C 停止服务器")
    
    # 鍒濆鍖栧苟杩愯鏈嶅姟鍣?    mcp.run(transport='stdio') 


from __future__ import annotations
import os, requests
from dataclasses import dataclass
from typing import Optional

@dataclass
class WeatherResult:
    city: str
    description: str
    temperature_c: float
    provider: str
    feels_like_c: Optional[float] = None
    humidity_pct: Optional[float] = None
    wind_kph: Optional[float] = None

def _fetch_openweathermap(city: str, api_key: str) -> WeatherResult:
    url = "https://api.openweathermap.org/data/2.5/weather"
    params = {"q": city, "appid": api_key, "units": "metric"}
    r = requests.get(url, params=params, timeout=15)
    r.raise_for_status()
    data = r.json()
    desc = data["weather"][0]["description"]
    main = data["main"]
    wind = data.get("wind", {})
    return WeatherResult(
        city=data.get("name", city),
        description=desc,
        temperature_c=float(main.get("temp")),
        feels_like_c=float(main.get("feels_like", main.get("temp"))),
        humidity_pct=float(main.get("humidity")) if main.get("humidity") is not None else None,
        wind_kph=float(wind.get("speed")) * 3.6 if wind.get("speed") is not None else None,  # m/s→km/h
        provider="openweathermap",
    )

def _geocode_city(city: str):
    geo_url = "https://geocoding-api.open-meteo.com/v1/search"
    r = requests.get(geo_url, params={"name": city, "count": 1}, timeout=15)
    r.raise_for_status()
    data = r.json()
    if not data.get("results"):
        raise ValueError(f"Could not geocode city: {city}")
    res = data["results"][0]
    return res["latitude"], res["longitude"], res.get("name", city)

def _fetch_open_meteo(city: str) -> WeatherResult:
    lat, lon, resolved_name = _geocode_city(city)
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "current": "temperature_2m,relative_humidity_2m,apparent_temperature,wind_speed_10m,weather_code",
    }
    r = requests.get(url, params=params, timeout=15)
    r.raise_for_status()
    data = r.json()
    cur = data.get("current", {})
    temp = cur.get("temperature_2m")
    feels = cur.get("apparent_temperature")
    rh = cur.get("relative_humidity_2m")
    wind = cur.get("wind_speed_10m")
    return WeatherResult(
        city=resolved_name,
        description="current conditions",
        temperature_c=float(temp) if temp is not None else 0.0,
        feels_like_c=float(feels) if feels is not None else None,
        humidity_pct=float(rh) if rh is not None else None,
        wind_kph=float(wind) if wind is not None else None,
        provider="open-meteo",
    )

def fetch_weather(city: str, api_key: str | None = None) -> WeatherResult:
    api_key = api_key or os.getenv("OPENWEATHERMAP_API_KEY")
    if api_key:
        try:
            return _fetch_openweathermap(city, api_key)
        except Exception:
            pass
    return _fetch_open_meteo(city)

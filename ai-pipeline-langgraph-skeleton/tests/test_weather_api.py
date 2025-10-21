import os, pytest
from src.weather.api import fetch_weather

@pytest.mark.skipif(not os.getenv("OPENWEATHERMAP_API_KEY"), reason="No OWM API key configured")
def test_fetch_weather_openweathermap():
    r = fetch_weather("Chennai")
    assert r.city
    assert isinstance(r.temperature_c, float)
    assert isinstance(r.description, str)
    assert r.provider in ("openweathermap", "open-meteo")

@pytest.mark.skipif(os.getenv("RUN_NETWORK_TESTS") != "1", reason="Network tests disabled")
def test_fetch_weather_open_meteo_fallback():
    # Ensure no OWM key to exercise fallback path
    os.environ.pop("OPENWEATHERMAP_API_KEY", None)
    r = fetch_weather("Chennai")
    assert r.city
    assert isinstance(r.temperature_c, float)
    assert r.provider == "open-meteo"

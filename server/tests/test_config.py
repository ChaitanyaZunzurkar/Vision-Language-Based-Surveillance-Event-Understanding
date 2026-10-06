"""Tests for configuration loader."""

from server.src.config.loader import config_loader


def test_config_loads_values():
    assert config_loader.get("server.port") == 8000
    assert config_loader.get("detection.conf_threshold") == 0.20
    assert config_loader.get("temporal.window_duration_sec") == 10.0


def test_classes_definition():
    classes = config_loader.detection_classes
    assert len(classes) > 0
    assert any(c["name"] == "Person" for c in classes)

    anomalies = config_loader.anomaly_categories
    assert len(anomalies) == 14
    assert any(a["name"] == "Burglary" for a in anomalies)
    assert any(a["name"] == "Normal" for a in anomalies)



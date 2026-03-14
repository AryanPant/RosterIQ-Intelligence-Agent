import schedule
import time

from engines.anomaly_detector import AnomalyDetector
from utils.config_loader import load_config


class MonitoringEngine:

    def __init__(self, detector=None, interval_minutes=None):

        self.detector = detector or AnomalyDetector()
        config = load_config("config/system_config.yaml")
        self.interval_minutes = interval_minutes or config["monitoring"].get("interval_minutes", 5)

    def monitor(self, market=None):

        anomalies = self.detector.detect(market=market)
        payload = {
            "detected_count": len(anomalies),
            "market": market or "ALL",
            "anomalies": anomalies,
        }
        print(payload)
        return payload

    def start(self, market=None):

        schedule.every(self.interval_minutes).minutes.do(self.monitor, market=market)
        self.monitor(market=market)

        while True:
            schedule.run_pending()
            time.sleep(10)

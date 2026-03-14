from tools.data_query_tool import DataQueryTool
from utils.config_loader import load_config


class AnomalyDetector:

    def __init__(self):

        self.data = DataQueryTool()
        config = load_config("config/system_config.yaml")
        self.rejection_threshold = config["analysis"]["rejection_threshold"]

    def detect(self, market=None):

        anomalies = []

        rejection_anomalies = self.data.detect_rejection_rate_anomalies(self.rejection_threshold)
        if market and not rejection_anomalies.empty:
            rejection_anomalies = rejection_anomalies[
                rejection_anomalies["CNT_STATE"].astype(str).str.upper() == market.upper()
            ]
        if not rejection_anomalies.empty:
            anomalies.append(
                {
                    "type": "high_rejection_rate_files",
                    "market": market or "ALL",
                    "count": int(len(rejection_anomalies)),
                }
            )

        duration_anomalies = self.data.detect_stage_duration_anomalies()
        if market and not duration_anomalies.empty:
            duration_anomalies = duration_anomalies[
                duration_anomalies["CNT_STATE"].astype(str).str.upper() == market.upper()
            ]
        if not duration_anomalies.empty:
            anomalies.append(
                {
                    "type": "stage_duration_outliers",
                    "market": market or "ALL",
                    "count": int(len(duration_anomalies)),
                }
            )

        return anomalies

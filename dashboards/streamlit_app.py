"""Streamlit view of the API prediction log and Evidently report."""
import logging

import pandas as pd
import streamlit as st

from meteorite.config import Settings
from meteorite.monitoring.drift import evaluate_drift, load_prediction_log, load_reference
from meteorite.prediction_log import PredictionLogger

logger = logging.getLogger(__name__)


def main() -> None:
    """Show logged prediction activity and feature drift."""
    st.set_page_config(page_title="Meteorite model monitoring", layout="wide")
    st.title("Meteorite model monitoring")
    st.button("Refresh data")

    try:
        predictions = load_prediction_log(PredictionLogger.from_env().path)
        if predictions is None:
            st.info("No predictions yet. Send a request to the API’s /predict endpoint and refresh.")
            return

        timestamps = pd.to_datetime(predictions["timestamp"], utc=True, errors="coerce", format="mixed")
        st.metric("Logged predictions", len(predictions))
        daily = timestamps.dropna().dt.strftime("%Y-%m-%d").value_counts().sort_index()
        if not daily.empty:
            st.bar_chart(daily.rename("Predictions"), x_label="Day (UTC)", y_label="Predictions", height=230)

        st.subheader("Feature drift")
        reference = load_reference(Settings.from_env().data_path)
        report = evaluate_drift(reference, predictions)
    except FileNotFoundError:
        st.error("The training sample is missing. Run dvc pull before starting the stack.")
        return
    except (KeyError, ValueError, OSError):
        logger.exception("Could not generate the drift report")
        st.error("The report could not be generated. Check the prediction CSV and the training sample.")
        return

    st.metric("Drifted features", report.summary["count"])
    if report.summary["columns"]:
        st.write(", ".join(report.summary["columns"]))
    if report.summary["skipped"]:
        st.info("Insufficient values in: " + ", ".join(report.summary["skipped"]))
    st.iframe(report.html(), height=1050, alt="Evidently data drift report")


if __name__ == "__main__":
    main()

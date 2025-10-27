import pandas as pd

from bid_predictor.ui import BAR_COLOR_SEQUENCE, build_prediction_plot


def test_build_prediction_plot_creates_traces():
    df = pd.DataFrame(
        {
            "Bid #": [1, 1, 2],
            "Acceptance Probability": [50.0, 60.0, 70.0],
            "current_timestamp": pd.to_datetime(
                ["2024-01-01", "2024-01-02", "2024-01-01"]
            ),
            "departure_timestamp": pd.to_datetime(
                ["2024-01-03", "2024-01-04", "2024-01-05"]
            ),
            "offer_status": ["pending", "accepted", "rejected"],
        }
    )

    fig = build_prediction_plot(df)
    assert fig.layout.title.text == "Acceptance probability by snapshot"
    assert len(fig.data) >= 2
    assert BAR_COLOR_SEQUENCE


def test_build_prediction_plot_handles_empty():
    fig = build_prediction_plot(pd.DataFrame())
    assert "No predictions available" in fig.layout.title.text

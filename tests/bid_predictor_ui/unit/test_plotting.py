import pandas as pd

from bid_predictor_ui import BAR_COLOR_SEQUENCE, build_prediction_plot


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


def test_prediction_plot_hover_includes_bid_number():
    df = pd.DataFrame(
        {
            "Bid #": [42, 42],
            "Acceptance Probability": [55.0, 65.0],
            "snapshot_num": [1, 2],
            "offer_status": ["pending", "accepted"],
        }
    )

    fig = build_prediction_plot(df)
    first_trace = fig.data[0]

    assert "Bid #: %{customdata" in first_trace.hovertemplate
    # Snapshot column should be present before bid values in customdata
    assert first_trace.customdata.shape[1] == 2
    assert first_trace.customdata[0, 1] == "42"
    assert "Snapshot:" in first_trace.hovertemplate

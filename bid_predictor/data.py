import pandas as pd
from .feature_config import _GROUPBY_KEY_FEATURES


def prepare_features(data, pre_features, testing=True):
    data = data[
        data.departure_timestamp - data.current_timestamp < pd.to_timedelta("5d")
    ]

    data.sort_values(["travel_date", "carrier_code", "flight_number"]).reset_index(
        drop=True
    )

    available_pre_features = [
        feature for feature in pre_features if feature in data.columns
    ]
    selection_columns = list(dict.fromkeys(available_pre_features + ["offer_status", "id", "decision_timestamp"]))

    if testing:
        cutoff = "2023-08-01"
        yX_test = data[
            (data.travel_date >= cutoff) & (data.travel_date <= "2023-08-15")
        ][selection_columns]
    else:
        cutoff = "2025-05-01"
        yX_test = data.loc[data.travel_date >= cutoff,selection_columns]
    yX_train = data.loc[data.travel_date < cutoff,selection_columns]

    X_train, X_test = yX_train.loc[:,available_pre_features], yX_test.loc[:,available_pre_features]
    y_train = (yX_train["offer_status"] == "TICKETED").astype(int)
    y_test = (yX_test["offer_status"] == "TICKETED").astype(int)

    yX_test.loc[:,"offer_status"] = "Rejected"
    yX_test.loc[y_test==1,"offer_status"] = "Accepted"

    yX_test = yX_test.set_index(_GROUPBY_KEY_FEATURES)# + ["current_timestamp"])
    # yX_test = yX_test.sort_index()

    yX_test["Bid #"] = (
        yX_test
        .groupby(level=_GROUPBY_KEY_FEATURES[:-1], observed=True)["id"]
        .transform(lambda s: pd.factorize(s)[0] + 1)
        .astype(int)
        .apply(lambda n: f"Bid {n}")
    )

    return X_train, X_test, y_train, y_test, yX_test
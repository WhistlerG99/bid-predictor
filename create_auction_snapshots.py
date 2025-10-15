import numpy as np
import pandas as pd


avail_cols = [
    "available_count_01h",
    "available_count_12h",
    "available_count_24h",
    "available_count_48h",
    "available_count_72h",
]

event_date_cols = [
    "event_local_date_01h",
    "event_local_date_12h",
    "event_local_date_24h",
    "event_local_date_48h",
    "event_local_date_72h",
]

pre_features = (
    [
        "id",
        "partner_id",
        "conf_num",
        "offer_status",
        "carrier_code",
        "flight_number",
        "travel_date",
        "item_count",
        "usd_base_amount",
        "fare_class",
        "offer_time",
        "multiplier_fare_class",
        "multiplier_loyalty",
        "multiplier_success_history",
        "multiplier_payment_type",
        "from_cabin",
        "upgrade_type",
    ]
    + ["decision_timestamp", "created"]
    + event_date_cols
    + avail_cols
)


features = [
    "id",
    "partner_id",
    "conf_num",
    "offer_status",
    "carrier_code",
    "flight_number",
    "travel_date",
    "item_count",
    "usd_base_amount",
    "fare_class",
    "offer_time",
    "multiplier_fare_class",
    "multiplier_loyalty",
    "multiplier_success_history",
    "multiplier_payment_type",
    "from_cabin",
    "upgrade_type",
    "decision_timestamp",
    "created",
    "current_timestamp",
    "current_available_seats",
    "current_seats_col",
    "snapshot_num",
]


def filter_by_date_with_availability(
    df: pd.DataFrame,
    current_date,
    created_col: str = "created",
    decision_timestamp_col: str = "decision_timestamp",
    event_date_cols: list = None,
    avail_cols: list = None,
    new_col: str = "current_available_seats",
) -> pd.DataFrame:
    """
    Filter rows of df by a given current_date and compute a current_available_seats value.

    Filtering rules:
    - If current_date is before 'created' -> row is filtered out.
    - If created <= current_date < decision_timestamp -> row is kept.
    - If current_date >= decision_timestamp -> row is filtered out.

    Availability rule (based on event_date_cols / avail_cols):
    - Consider the event_local_date_* columns in ordinal order (oldest to newest):
      event_local_date_01h, event_local_date_12h, event_local_date_24h,
      event_local_date_48h, event_local_date_72h
    - If current_date is before event_local_date_72h, use available_count_72h.
    - Otherwise, find the latest event_local_date_* that is <= current_date
      and use the corresponding available_count_* value.
    - If a matching available_count_* column is missing, result will be NaN for that row.

    Parameters:
    - df: input DataFrame
    - current_date: date against which to filter and compute availability
    - created_col: name of the 'created' column
    - decision_timestamp_col: name of the 'decision_timestamp' column
    - event_date_cols: list of event date column names in chronological order
                       (defaults to a standard set if None)
    - avail_cols: list of corresponding available_count column names in same order
                  as event_date_cols
    - new_col: name of the output column to store current availability

    Returns:
    - DataFrame filtered by the date rules with an added column 'new_col'
    """
    # Normalize current_date to Timestamp
    current_ts = pd.to_datetime(current_date)

    if created_col not in df.columns:
        raise KeyError(f"Column '{created_col}' not found in DataFrame.")
    if decision_timestamp_col not in df.columns:
        raise KeyError(f"Column '{decision_timestamp_col}' not found in DataFrame.")

    if event_date_cols is None:
        event_date_cols = [
            "event_local_date_01h",
            "event_local_date_12h",
            "event_local_date_24h",
            "event_local_date_48h",
            "event_local_date_72h",
        ][::-1]
    if avail_cols is None:
        avail_cols = [
            "available_count_01h",
            "available_count_12h",
            "available_count_24h",
            "available_count_48h",
            "available_count_72h",
        ][::-1]

    # Basic mask for filtering
    created_ts = pd.to_datetime(df[created_col])
    decision_ts = pd.to_datetime(df[decision_timestamp_col])

    mask = (created_ts <= current_ts) & (decision_ts > current_ts)

    # Function to pick the appropriate availability for a single row
    def pick_availability(row: pd.Series) -> np.float64:
        # If any of the event columns are missing, attempt to proceed with available_count_72h as a fallback
        # Build list of (event_date, avail_col) in chronological order
        for i, ev_col in enumerate(event_date_cols):
            if ev_col not in row or pd.isna(row.get(ev_col)):
                continue
            ev_ts = pd.to_datetime(row[ev_col])
            if current_ts < ev_ts:
                # current is before this event date; use the previous threshold's available count
                if i == 0:
                    # before the first event, use the last (72h) as per requirement
                    return current_ts, row.get(avail_cols[0], np.nan), avail_cols[0]
                else:
                    return (
                        current_ts,
                        row.get(avail_cols[i - 1], np.nan),
                        avail_cols[i - 1],
                    )
        # If we reach here, current_ts is after or equal to the last event date
        return current_ts, row.get(avail_cols[0], np.nan), avail_cols[0]

    # Compute current_available_seats per row
    df = df.copy()
    df[["current_timestamp", new_col, "current_seats_col"]] = df.apply(
        pick_availability, axis=1, result_type="expand"
    )

    # Return only rows that satisfy the mask, with the new column in place
    return df.loc[mask]


def timestamps_between(ts: pd.Series) -> pd.Series:
    """
    Given a 1D array-like of timestamps, return a Series of midpoints
    between consecutive timestamps (sorted ascending).
    Result length will be len(ts) - 1.
    """
    ts = pd.to_datetime(ts).sort_values().reset_index(drop=True)
    if ts.size < 2:
        return pd.Series([], dtype="datetime64[ns]")
    mids = [ts.iloc[i] + (ts.iloc[i + 1] - ts.iloc[i]) / 2 for i in range(ts.size - 1)]
    return pd.Series(mids)


def event_times(df):
    decision_timestamps = df[["decision_timestamp"]].drop_duplicates()
    decision_timestamps.index = [
        f"decision_timestamp{i+1}" for i in range(len(decision_timestamps.index))
    ]
    decision_timestamps.columns = [0]

    createds = df[["created"]].drop_duplicates()
    createds.index = [f"created{i+1}" for i in range(len(createds.index))]
    createds.columns = [0]

    event_dates = df[event_date_cols].drop_duplicates().T
    event_dates.columns = [0]

    times = pd.concat((decision_timestamps, createds, event_dates))
    
    try:
        times = pd.to_datetime(times[0], format='ISO8601')
    except ValueError as e:
        print(times)
        print(e)
    times = times.dropna().sort_values()

    start_col, start_time = (
        times.drop(index=[x for x in times.index if x.startswith("created")])
        .reset_index()
        .iloc[0]
        .tolist()
    )
    end_time = times[times.index.str.startswith("decision_timestamp")].iloc[-1]

    # for ev_col in event_date_cols[::-1]:
    #     if ev_col in df and not pd.isna(df[ev_col].iloc[0]):
    #         start_time = df[ev_col].iloc[0]
    #         start_col = ev_col
    #         break

    # times = times[times >= start_time].dropna()
    times = times[(times >= start_time) & (times <= end_time)].dropna()

    times["start_time"] = times[start_col] - pd.to_timedelta("2 day")
    times = times.sort_values()

    if times[times.index.str.contains("decision_timestamp")].size == 0:
        times = pd.Series(name=0, dtype=np.dtype("<M8[ns]"))
    return times


def get_snapshots(df):
    dt_index = timestamps_between(event_times(df)).tolist()
    # start = round_down_to_next_6h(pd.to_datetime(df["event_local_date_72h"].iloc[0]))
    # end = round_up_to_next_6h(df["decision_timestamp"].max())
    # dt_index = pd.date_range(start=start, end=end, freq="1h")

    dfs = []
    i = 1
    for dt in dt_index:
        _df = filter_by_date_with_availability(df, dt)
        if (
            len(dfs) == 0
            or not dfs[-1]
            .drop(columns=["snapshot_num", "current_timestamp"])
            .equals(_df.drop(columns=["current_timestamp"]))
        ) and _df.shape[0] > 0:
            _df["snapshot_num"] = i
            i += 1
            dfs.append(_df)
    if len(dfs) > 0:
        dfs = pd.concat(dfs)
    else:
        dfs = pd.DataFrame(
            columns=df.columns.tolist()
            + [
                "current_timestamp",
                "current_available_seats",
                "current_seats_col",
                "snapshot_num",
            ]
        )
    return dfs


if __name__ == "__main__":
    train_file = "~/Documents/old-bid-predictor/bid_data_enriched_new.csv"

    data = pd.read_csv(
        train_file,
        parse_dates=["travel_date","decision_timestamp","created"] + event_date_cols,
        dtype={
            "carrier_code": "category",
            "flight_number": "category",
            "fare_class": "category",
        },
        low_memory=False,
    )

    data = data.sort_values(
        ["travel_date", "carrier_code", "flight_number"]
    ).reset_index(drop=True)[pre_features]

    data = data[data.travel_date >= "2025-05-01"]

    id_cnts = data.groupby("id").size().sort_values()
    duplicate_ids = id_cnts[id_cnts > 1].index  # ids with count > 1
    data = data[~data["id"].isin(duplicate_ids)]

    data = data.groupby(
        ["carrier_code", "flight_number", "travel_date", "upgrade_type"], observed=True
    ).apply(get_snapshots, include_groups=False)

    data = data.reset_index()[features]

    data.to_csv("./bid_data_snapshots.csv", index=False)

#!/usr/bin/env python3
"""
PySpark rewrite of create_auction_snapshots.py using a grouped-map Pandas UDF
so snapshot generation runs in parallel across partitions/groups.

Designed for AWS SageMaker / EMR: reads from S3, distributes per-flight groups,
and writes snapshots back to S3 (or local) as a single CSV.

Usage (example):
    spark-submit --deploy-mode client create_auction_snapshots_pyspark.py \
        --input s3://your-bucket/path/bid_data_enriched_new.csv \
        --output s3://your-bucket/path/bid_data_snapshots.csv

Notes:
- Uses Arrow-accelerated applyInPandas for parallelism across groups.
- Keeps the original pandas logic for snapshot timing / availability selection.
- If your input timestamps are strings, they will be parsed inside the UDF.
"""

from __future__ import annotations

import argparse
import numpy as np
import pandas as pd
from typing import List

from pyspark.sql import SparkSession, functions as F, types as T

# -----------------------------
# Column sets (kept from original)
# -----------------------------
avail_cols: List[str] = [
    "available_count_01h",
    "available_count_12h",
    "available_count_24h",
    "available_count_48h",
    "available_count_72h",
]

event_date_cols: List[str] = [
    "event_local_date_01h",
    "event_local_date_12h",
    "event_local_date_24h",
    "event_local_date_48h",
    "event_local_date_72h",
]

# Final output columns (order preserved)
features: List[str] = [
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

# -----------------------------
# Pandas helpers (ported as-is)
# -----------------------------


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


def event_times(df: pd.DataFrame) -> pd.DataFrame:
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
    times = pd.to_datetime(times[0], format='ISO8601')
    times = times.dropna().sort_values()

    # times_ = times.drop(index=[x for x in times.index if x.startswith("created")])
    # if times_.size==0:
    #     return pd.Series(name=0, dtype=np.dtype("<M8[ns]"))
    # start_col, start_time = times_.reset_index().iloc[0].tolist()

    try:
        start_col, start_time = (
            times.drop(index=[x for x in times.index if x.startswith("created")])
            .reset_index()
            .iloc[0]
            .tolist()
        )
    except IndexError:
        key = df[["carrier_code", "flight_number", "travel_date", "upgrade_type"]].iloc[0].values.tolist()
        raise IndexError(f"1) This is the problem dataframe: {str(times)},\nThis is the flight {str(key)}")

    times_ = times[times.index.str.startswith("decision_timestamp")]
    if times_.size==0:
        return pd.Series(name=0, dtype=np.dtype("<M8[ns]"))
    end_time = times_.iloc[-1]

    times = times[(times >= start_time) & (times <= end_time)].dropna()

    times["start_time"] = times[start_col] - pd.to_timedelta("2 day")
    times = times.sort_values()

    if times[times.index.str.contains("decision_timestamp")].size == 0:
        times = pd.Series(name=0, dtype=np.dtype("<M8[ns]"))
    return times


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
            "event_local_date_72h",
            "event_local_date_48h",
            "event_local_date_24h",
            "event_local_date_12h",
            "event_local_date_01h",
        ]
    if avail_cols is None:
        avail_cols = [
            "available_count_72h",
            "available_count_48h",
            "available_count_24h",
            "available_count_12h",
            "available_count_01h",
        ]

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


def get_snapshots(df: pd.DataFrame) -> pd.DataFrame:
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

    keep = [c for c in features if c in dfs.columns]
    return dfs[keep]

# -----------------------------
# Spark job
# -----------------------------


def build_spark(app_name: str = "auction-snapshots") -> SparkSession:
    spark = (
        SparkSession.builder.appName(app_name)
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .getOrCreate()
    )
    return spark


def output_schema() -> T.StructType:
    return T.StructType([
        T.StructField("id", T.LongType(), True),
        T.StructField("partner_id", T.IntegerType(), True),
        T.StructField("conf_num", T.StringType(), True),
        T.StructField("offer_status", T.StringType(), True),
        T.StructField("carrier_code", T.StringType(), True),
        T.StructField("flight_number", T.StringType(), True),
        T.StructField("travel_date", T.TimestampType(), True),
        T.StructField("item_count", T.IntegerType(), True),
        T.StructField("usd_base_amount", T.DoubleType(), True),
        T.StructField("fare_class", T.StringType(), True),
        T.StructField("offer_time", T.DoubleType(), True),
        T.StructField("multiplier_fare_class", T.DoubleType(), True),
        T.StructField("multiplier_loyalty", T.DoubleType(), True),
        T.StructField("multiplier_success_history", T.DoubleType(), True),
        T.StructField("multiplier_payment_type", T.DoubleType(), True),
        T.StructField("from_cabin", T.StringType(), True),
        T.StructField("upgrade_type", T.StringType(), True),
        T.StructField("decision_timestamp", T.TimestampType(), True),
        T.StructField("created", T.TimestampType(), True),
        T.StructField("current_timestamp", T.TimestampType(), True),
        T.StructField("current_available_seats", T.IntegerType(), True),
        T.StructField("current_seats_col", T.StringType(), True),
        T.StructField("snapshot_num", T.IntegerType(), True),
    ])


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="Input CSV path (S3 or local)")
    p.add_argument("--output", required=True, help="Output CSV path (S3 or local)")
    p.add_argument(
        "--min_travel_date",
        default="2025-05-01",
        help="Filter travel_date >= this (YYYY-MM-DD)",
    )
    return p.parse_args()


def main():
    args = parse_args()
    spark = build_spark()

    # Read CSV. Schema is inferred; if you know exact schema, set it explicitly
    df = (
        spark.read.option("header", True)
        .option("inferSchema", True)
        .csv(args.input)
    )
    df = df.drop(*[c for c in df.columns if "." in c])
    df = df[pre_features]
    # Basic hygiene: drop exact duplicate rows
    df = df.dropDuplicates()

    # Normalize primitive columns to avoid nested/struct misreads from some sources
    for c, typ in [
        ("id", "long"),
        ("partner_id", "int"),
        ("item_count", "int"),
        ("usd_base_amount", "double"),
        ("flight_number", "string"),
    ]:
        if c in df.columns:
            df = df.withColumn(c, F.col(c).cast(typ))

    # Filter travel_date >= cutoff (cast if it came as string)
    df = df.withColumn(
        "travel_date",
        F.to_timestamp("travel_date")
    )
    if args.min_travel_date is not None:
        df = df.filter(F.col("travel_date") >= F.lit(args.min_travel_date))

    # Remove duplicate ids (ids with count > 1), like the pandas version
    id_cnts = df.groupBy("id").count()
    dup_ids = id_cnts.filter(F.col("count") > 1).select("id")
    df = df.join(dup_ids, on="id", how="left_anti")

    # Group keys (same as original pandas groupby)
    group_keys = ["carrier_code", "flight_number", "travel_date", "upgrade_type"]

    # Ensure columns used by pandas helpers exist; cast likely timestamp fields to string so pandas can parse flexibly
    for c in ["decision_timestamp", "created", *event_date_cols]:
        if c in df.columns:
            df = df.withColumn(c, F.col(c).cast("string"))

    # Run the grouped-map Pandas UDF in parallel across groups
    def grouped_snapshots(pdf: pd.DataFrame) -> pd.DataFrame:
        # Ensure expected columns are present even if missing in the partition
        for c in event_date_cols + avail_cols:
            if c not in pdf.columns:
                pdf[c] = np.nan
        out = get_snapshots(pdf)
        # Enforce dtypes that match output_schema to avoid parser confusion
        for c in ["id", "partner_id", "item_count", "current_available_seats", "snapshot_num"]:
            if c in out.columns:
                out[c] = pd.to_numeric(out[c], errors="coerce").astype("Int64")
        for c in ["decision_timestamp", "created", *event_date_cols]:
            if c in out.columns:
                out[c] = pd.to_datetime(out[c], errors="coerce")
        if "usd_base_amount" in out.columns:
            out["usd_base_amount"] = pd.to_numeric(out["usd_base_amount"], errors="coerce")
        return out

    result = (
        df.groupBy(group_keys)
        .applyInPandas(grouped_snapshots, schema=output_schema())
    )

    # Reorder/select to output features and write a single CSV (coalesce(1) to avoid sharded output)
    # NOTE: For large outputs, you may want to keep it sharded and downstream-consolidate.
    # Use explicit col() selection to avoid Spark trying to extract nested fields from scalar INTs
    result = result.select(*[F.col(c) for c in features if c in result.columns])

    result.write.mode("overwrite").parquet(args.output)

    spark.stop()


if __name__ == "__main__":
    main()

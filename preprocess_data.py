from bid_predictor.preprocessor import (
    load_flight_data,
    load_offer_data,
    preprocess_data,
    get_seats_available,
)


if __name__ == "__main__":
    df_flights = load_flight_data(
        "~/data/bid_predictor/availability_and_flight_metadata_ACA_LOT_202509220859.csv"
    )
    df_offers = load_offer_data("~/data/bid_predictor/joined_data.csv")
    data = preprocess_data(df_flights, df_offers)
    data["seats_available"] = data.apply(get_seats_available, axis=1)
    data.to_csv("bid_data_enriched_new.csv", index=False)

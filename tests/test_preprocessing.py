# Import libraries
from decimal import Decimal

import pandas as pd

from main import TripInput, build_input_dataframe, to_decimal


def test_build_input_dataframe_returns_single_row():
    trip = TripInput(
        trip_distance=3.5,
        passenger_count=2,
        PULocationID=142,
        DOLocationID=236,
        RatecodeID=1,
    )

    df = build_input_dataframe(trip)

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 1


def test_build_input_dataframe_has_expected_columns_in_order():
    trip = TripInput(
        trip_distance=3.5,
        passenger_count=2,
        PULocationID=142,
        DOLocationID=236,
        RatecodeID=1,
    )

    df = build_input_dataframe(trip)

    assert list(df.columns) == [
        "trip_distance",
        "passenger_count",
        "PULocationID",
        "DOLocationID",
        "RatecodeID",
    ]


def test_build_input_dataframe_preserves_values():
    trip = TripInput(
        trip_distance=5.25,
        passenger_count=3,
        PULocationID=100,
        DOLocationID=200,
        RatecodeID=2,
    )

    df = build_input_dataframe(trip)

    assert df.iloc[0]["trip_distance"] == 5.25
    assert df.iloc[0]["passenger_count"] == 3
    assert df.iloc[0]["PULocationID"] == 100
    assert df.iloc[0]["DOLocationID"] == 200
    assert df.iloc[0]["RatecodeID"] == 2


def test_to_decimal_converts_float_to_decimal_type():
    result = to_decimal(18.42)
    assert isinstance(result, Decimal)


def test_to_decimal_preserves_value_accurately():
    # Going through str() avoids binary float artifacts, e.g. a naive
    # Decimal(0.1) conversion would NOT equal Decimal("0.1").
    result = to_decimal(0.1)
    assert result == Decimal("0.1")


def test_to_decimal_handles_integers_passed_as_float():
    result = to_decimal(20.0)
    assert result == Decimal("20.0")

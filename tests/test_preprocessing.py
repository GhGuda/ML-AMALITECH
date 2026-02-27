"""Unit tests for Stage 2 cleaning and preprocessing logic."""

from __future__ import annotations

import unittest

import pandas as pd

from src.flight_fare.preprocessing import clean_and_engineer_features, split_and_preprocess


def _raw_sample_dataframe() -> pd.DataFrame:
    """Create a dataframe that includes common data-quality issues."""
    return pd.DataFrame(
        [
            {
                "Airline": "Airline A",
                "Source": "dac ",
                "Source Name": "dacca",
                "Destination": "cgp",
                "Destination Name": "chittagong",
                "Departure Date & Time": "2025-01-01 10:00:00",
                "Arrival Date & Time": "2025-01-01 11:00:00",
                "Duration (hrs)": "1.0",
                "Stopovers": "0",
                "Aircraft Type": "A320",
                "Class": "Economy",
                "Booking Source": "Web",
                "Base Fare (BDT)": "1000",
                "Tax & Surcharge (BDT)": "200",
                "Total Fare (BDT)": "1200",
                "Seasonality": "Winter",
                "Days Before Departure": "10",
                "Unnamed: 0": 1,
            },
            {
                "Airline": "Airline A",
                "Source": "dac ",
                "Source Name": "dacca",
                "Destination": "cgp",
                "Destination Name": "chittagong",
                "Departure Date & Time": "2025-01-01 10:00:00",
                "Arrival Date & Time": "2025-01-01 11:00:00",
                "Duration (hrs)": "1.0",
                "Stopovers": "0",
                "Aircraft Type": "A320",
                "Class": "Economy",
                "Booking Source": "Web",
                "Base Fare (BDT)": "1000",
                "Tax & Surcharge (BDT)": "200",
                "Total Fare (BDT)": "1200",
                "Seasonality": "Winter",
                "Days Before Departure": "10",
                "Unnamed: 0": 1,
            },
            {
                "Airline": "Airline B",
                "Source": "jsr",
                "Source Name": "Jessore",
                "Destination": "zyl",
                "Destination Name": "sylhet ",
                "Departure Date & Time": "2025-06-12 07:30:00",
                "Arrival Date & Time": "2025-06-12 08:55:00",
                "Duration (hrs)": "1.2",
                "Stopovers": "1",
                "Aircraft Type": "B737",
                "Class": "Economy",
                "Booking Source": "App",
                "Base Fare (BDT)": "-100",
                "Tax & Surcharge (BDT)": "350",
                "Total Fare (BDT)": "250",
                "Seasonality": "Summer",
                "Days Before Departure": "5",
                "Unnamed: 0": 2,
            },
            {
                "Airline": "Airline C",
                "Source": "DAC",
                "Source Name": "Dhaka",
                "Destination": "CXB",
                "Destination Name": "Coxs Bazar",
                "Departure Date & Time": "invalid-date",
                "Arrival Date & Time": "2025-08-20 15:30:00",
                "Duration (hrs)": "1.5",
                "Stopovers": "0",
                "Aircraft Type": "A321",
                "Class": "Business",
                "Booking Source": "Web",
                "Base Fare (BDT)": "3000",
                "Tax & Surcharge (BDT)": "900",
                "Total Fare (BDT)": "3900",
                "Seasonality": "Monsoon",
                "Days Before Departure": "14",
                "Unnamed: 0": 3,
            },
            {
                "Airline": "Airline D",
                "Source": "RJH",
                "Source Name": "Rajshahi",
                "Destination": "DAC",
                "Destination Name": "Dhaka",
                "Departure Date & Time": "2025-11-02 05:20:00",
                "Arrival Date & Time": "2025-11-02 06:30:00",
                "Duration (hrs)": "1.0",
                "Stopovers": "0",
                "Aircraft Type": "ATR72",
                "Class": "Economy",
                "Booking Source": "Agency",
                "Base Fare (BDT)": "1600",
                "Tax & Surcharge (BDT)": "300",
                "Total Fare (BDT)": "1900",
                "Seasonality": "Autumn",
                "Days Before Departure": "2",
                "Unnamed: 0": 4,
            },
            {
                "Airline": "Airline E",
                "Source": "CGP",
                "Source Name": "Chittagong",
                "Destination": "DAC",
                "Destination Name": "Dacca",
                "Departure Date & Time": "2025-03-18 09:40:00",
                "Arrival Date & Time": "2025-03-18 10:45:00",
                "Duration (hrs)": "1.1",
                "Stopovers": "0",
                "Aircraft Type": "A320",
                "Class": "Economy",
                "Booking Source": "Web",
                "Base Fare (BDT)": "1800",
                "Tax & Surcharge (BDT)": "450",
                "Total Fare (BDT)": "2250",
                "Seasonality": "Spring",
                "Days Before Departure": "7",
                "Unnamed: 0": 5,
            },
            {
                "Airline": "Airline F",
                "Source": "CXB",
                "Source Name": "Cox's bazar",
                "Destination": "DAC",
                "Destination Name": "Dhaka",
                "Departure Date & Time": "2025-04-07 22:10:00",
                "Arrival Date & Time": "2025-04-07 23:15:00",
                "Duration (hrs)": "1.1",
                "Stopovers": "0",
                "Aircraft Type": "B737",
                "Class": "Economy",
                "Booking Source": "App",
                "Base Fare (BDT)": "1700",
                "Tax & Surcharge (BDT)": "400",
                "Total Fare (BDT)": "2100",
                "Seasonality": "Spring",
                "Days Before Departure": "6",
                "Unnamed: 0": 6,
            },
        ]
    )


class PreprocessingTests(unittest.TestCase):
    """Test suite for cleaning and preprocessing rules."""

    def test_clean_and_engineer_features(self) -> None:
        """Cleaning should normalize values and add expected temporal features."""
        cleaned, report = clean_and_engineer_features(_raw_sample_dataframe())

        self.assertNotIn("Unnamed: 0", cleaned.columns)
        self.assertTrue({"Departure Month", "Departure Day", "Departure Weekday", "Departure Season"}.issubset(cleaned.columns))
        self.assertEqual(report.dropped_duplicates, 1)
        self.assertGreaterEqual(report.dropped_invalid_datetime_rows, 1)
        self.assertEqual(cleaned["Source"].str.upper().tolist(), cleaned["Source"].tolist())
        self.assertNotIn("Base Fare (BDT)", cleaned.columns)
        self.assertNotIn("Tax & Surcharge (BDT)", cleaned.columns)
        self.assertIn("Base Fare (BDT)", report.dropped_leaky_columns)
        self.assertIn("Tax & Surcharge (BDT)", report.dropped_leaky_columns)
        self.assertIn("Dhaka", cleaned["Source Name"].dropna().unique().tolist())
        self.assertIn("Chattogram", cleaned["Destination Name"].dropna().unique().tolist())

    def test_split_and_preprocess(self) -> None:
        """Split and preprocessing should return aligned train/test outputs."""
        cleaned, _ = clean_and_engineer_features(_raw_sample_dataframe())
        split = split_and_preprocess(dataframe=cleaned, test_size=0.4, random_state=42)

        self.assertEqual(len(split.x_train_processed), len(split.y_train))
        self.assertEqual(len(split.x_test_processed), len(split.y_test))
        self.assertGreater(split.x_train_processed.shape[1], 0)
        self.assertFalse(split.x_train_processed.isna().any().any())
        self.assertFalse(split.x_test_processed.isna().any().any())


if __name__ == "__main__":
    unittest.main()

import pandas as pd
from pathlib import Path

DATA_DIR = (Path(__file__).parent / "../data").resolve()
AIRNOW_AQS_CSV = DATA_DIR / "AirNowAQS/AirNow_AQS_Joined.csv"
OPENMETEO_CSV = DATA_DIR / "AirNow_OpenMeteo_merged_Jan2020-Oct2025.csv"
OUTPUT = DATA_DIR / "CompleteData/AirNow_AQS_Meteo_Joined_Jan2020-Oct2025.csv"


def main():

    print("Loading Airnow AQS data...")
    aqs_df = pd.read_csv(AIRNOW_AQS_CSV, dtype=str)

    print("Loading OpenMeteo data...")
    meteo_df = pd.read_csv(OPENMETEO_CSV, dtype=str)
    print("Loading complete.")

    print("Starting merge...")
    aqs_cols = aqs_df.columns.tolist()
    meteo_cols = meteo_df.columns.tolist()

    airnow_cols = aqs_cols[:10]
    aqs_only_cols = aqs_cols[10:]
    weather_only_cols = meteo_cols[10:]

    meteo_deduped = meteo_df.drop_duplicates(subset=airnow_cols, keep='first')

    merged = aqs_df.merge(
        meteo_deduped,
        on=airnow_cols,
        how='left',
        suffixes=('', '_meteo')
    )

    final_cols = airnow_cols + weather_only_cols + aqs_only_cols
    merged = merged[final_cols]
    print(f"Merged AirNow, OpenMeteo, AQS data resulting to {len(merged)} rows")

    print(f"Writing to {OUTPUT}...")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(OUTPUT, index=False)

    print(f"Done.")


if __name__ == "__main__":
    main()

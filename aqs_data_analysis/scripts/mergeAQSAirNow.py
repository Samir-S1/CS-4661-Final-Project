import pandas as pd
from pathlib import Path

DATA_DIR = (Path(__file__).parent / "../data").resolve()
AQS_CSV = DATA_DIR / "AQS/aqs_sites.csv"
AIRNOW_CSV = DATA_DIR / "AirNowAPI_responses/AirNowAPI_Jan2020-Oct2025_pm25_pm10_ozone_co_no2_so2.csv"
OUTPUT = DATA_DIR / "AirNowAQS/AirNow_AQS_Joined.csv"







def load_aqs():
    print("Loading AQS sites...")

    df = pd.read_csv(AQS_CSV, dtype=str)

    df['site_id'] = (
        df['State Code'].str.zfill(2) +
        df['County Code'].str.zfill(3) +
        df['Site Number'].str.zfill(4)
    )

    print(f"AQS sites loaded: {len(df)} rows")
    return df

def load_airnow():
    print("Loading AirNow data...")

    df = pd.read_csv(AIRNOW_CSV, dtype=str)

    df['site_id_norm'] = df[df.columns[8]].str.zfill(9)

    print(f"AirNow data loaded: {len(df)} rows")
    return df








def main():
    aqs_df = load_aqs()
    airnow_df = load_airnow()

    print("Merging datasets...")

    merged = airnow_df.merge(
        aqs_df,
        left_on='site_id_norm',
        right_on='site_id',
        how='left',
        suffixes=('', '_aqs')
    )

    merged = merged.drop(columns=['site_id_norm', 'site_id'])

    print(f"Merged dataset: {len(merged)} rows")

    print(f"Writing to {OUTPUT}...")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(OUTPUT, index=False)

    print(f"Output written to: {OUTPUT}")


if __name__ == "__main__":
    main()

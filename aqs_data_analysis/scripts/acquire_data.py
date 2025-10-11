import sqlite3
import pandas as pd
import requests
import zipfile
import io
from pathlib import Path

# Configuration
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_DIR = PROJECT_ROOT / 'data'
RAW_DATA_DIR = DATA_DIR / 'raw'
DATABASE_PATH = DATA_DIR / 'aqs_data.db'

AQS_BASE_URL = "https://aqs.epa.gov/aqsweb/airdata"

# Data files to acquire
# List of downloadable files: https://aqs.epa.gov/aqsweb/airdata/download_files.html#AQI
DATA_FILES = {
    'hourly_88101_2020': {
        'url': f'{AQS_BASE_URL}/hourly_88101_2020.zip',
    },
    'daily_44201_2025': {
        'url': f'{AQS_BASE_URL}/daily_44201_2025.zip',
    }
}


class AQSDataAcquisition:
    def __init__(self, db_path, data_dir, raw_dir):
        self.db_path = Path(db_path)
        self.data_dir = Path(data_dir)
        self.raw_dir = Path(raw_dir)

        self.data_dir.mkdir(exist_ok=True)
        self.raw_dir.mkdir(exist_ok=True)

    def download_file(self, url, file_key):
        print(f"\n{'='*60}")
        print(f"Downloading: {file_key}")
        print(f"URL: {url}")
        print(f"{'='*60}")

        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            zip_content = io.BytesIO(response.content)

            print("\nExtracting...")
            with zipfile.ZipFile(zip_content) as zf:
                csv_files = [f for f in zf.namelist() if f.endswith('.csv')]
                if csv_files:
                    csv_file = csv_files[0]
                    output_path = self.raw_dir / csv_file
                    with zf.open(csv_file) as source, open(output_path, 'wb') as target:
                        target.write(source.read())

                    print(f"Extracted: {csv_file}")
                    return {'success': True, 'local_path': str(output_path)}

            return {'success': False, 'error': 'No CSV file found in archive'}

        except requests.exceptions.RequestException as e:
            print(f"Download failed: {e}")
            return {'success': False, 'error': str(e)}

    def load_csv_to_table(self, csv_path, table_name):
        print(f"\nLoading into database: {table_name}")
        conn = sqlite3.connect(self.db_path)
        
        chunk_size = 100000  
        total_rows = 0
        
        for chunk in pd.read_csv(csv_path, chunksize=chunk_size, dtype=str):
            chunk.to_sql(table_name, conn, if_exists='append', index=False)
            total_rows += len(chunk)
        
        conn.close()
        print(f"\nCompleted: {total_rows:,} rows")


def main():
    print("="*60)
    print("AQS Data Acquisition & Integration")
    print("="*60)
    print(f"Database: {DATABASE_PATH}")
    print(f"Raw Data: {RAW_DATA_DIR}")
    print("="*60)

    acquisition = AQSDataAcquisition(DATABASE_PATH, DATA_DIR, RAW_DATA_DIR)

    print("\nConfiguration:")
    print(f"  Data files configured: {len(DATA_FILES)}")

    choice = input(
        "\nOptions:\n1. Download and load all data files\n2. Download only (no loading)\n3. Load existing files only\n\nSelect 1-3: "
    ).strip()

    if choice == '1':
        for file_key, file_info in DATA_FILES.items():
            print(f"\n{'='*60}")
            print(f"Processing: {file_key}")
            print(f"{'='*60}")

            result = acquisition.download_file(file_info['url'], file_key)
            if result['success']:
                csv_path = Path(result['local_path'])
                table_name = csv_path.stem
                acquisition.load_csv_to_table(csv_path, table_name)

    elif choice == '2':
        for file_key, file_info in DATA_FILES.items():
            acquisition.download_file(file_info['url'], file_key)

    elif choice == '3':
        for csv_file in RAW_DATA_DIR.glob('*.csv'):
            print(f"\nProcessing: {csv_file.name}")
            table_name = csv_file.stem
            acquisition.load_csv_to_table(csv_file, table_name)

    print("\n" + "="*60)
    print("Data Acquisition Complete")


if __name__ == "__main__":
    main()
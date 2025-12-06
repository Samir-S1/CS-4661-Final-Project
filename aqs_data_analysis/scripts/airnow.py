#remember to add .env file with AIRNOW_API_KEY variable in .env variable

import os
from dotenv import load_dotenv
import requests 
import pandas as pd  
import time  
import sys
import logging 
from datetime import datetime

# Load environment variables from .env file
load_dotenv()

# Get the API key from environment variables
AIRNOW_API_KEY = os.getenv('AIRNOW_API_KEY')

if AIRNOW_API_KEY is None:
    raise ValueError("AIRNOW_API_KEY not found in .env file. Please set it.")

print("AirNow API key loaded successfully.")


# File name to put data in
# API request URL
def build_request_url(options):
    return (options["url"]
            + "?startdate=" + options["start_date"]
            + "t" + options["start_hour_utc"]
            + "&enddate=" + options["end_date"]
            + "t" + options["end_hour_utc"]
            + "&parameters=" + options["parameters"]
            + "&bbox=" + options["bbox"]
            + "&datatype=" + options["data_type"]
            + "&format=" + options["format"]
            + "&verbose=" + options["verbose"]
            + "&monitorType=" + options["monitor_type"]
            + "&includeRawConcentration=" + options["include_raw_concentration"]
            + "&api_key=" + options["api_key"])





def step_request_date_range(options, max_datetime):
    current_start_date = datetime.strptime(options["start_date"] + "T" + options["start_hour_utc"], "%Y-%m-%dT%H")  
    current_end_date = datetime.strptime(options["end_date"] + "T" + options["end_hour_utc"], "%Y-%m-%dT%H")
    datetime_range = current_end_date - current_start_date
    max_datetime = datetime.strptime(max_datetime, "%Y-%m-%dT%H")

    new_start_date = current_end_date
    new_end_date = new_start_date + datetime_range

    if new_end_date > max_datetime:
        new_end_date = max_datetime
        at_max = True
    else:
        at_max = False

    options["start_date"] = new_start_date.strftime("%Y-%m-%d")
    options["start_hour_utc"] = new_start_date.strftime("%H")
    options["end_date"] = new_end_date.strftime("%Y-%m-%d")
    options["end_hour_utc"] = new_end_date.strftime("%H")
    return options, at_max


INITIAL_START_DATE = "2020-01-01"
INITIAL_START_HOUR_UTC = "00"
INITIAL_END_DATE = "2020-01-05"
INITIAL_END_HOUR_UTC = "00"

MAX_DATETIME = "2025-11-01T00"  



REQUEST_LIMIT = 500
TIME_WINDOW = 3600   
request_count = 0
hour_start_time = None

 
def main():
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()] 
    )

    # API parameters (initial range)
    options = {
        "url": "https://airnowapi.org/aq/data/",
        "start_date": INITIAL_START_DATE,
        "start_hour_utc": INITIAL_START_HOUR_UTC,
        "end_date": INITIAL_END_DATE,
        "end_hour_utc": INITIAL_END_HOUR_UTC,
        # Adjust this as needed
        "parameters": "pm25,pm10,ozone,co,no2,so2",
        "bbox": "-119.816581,32.571987,-115.158378,36.762816",
        "data_type": "c",
        "format": "text/csv",
        "ext": "csv",
        "verbose": "1",
        "monitor_type": "0",
        "include_raw_concentration": "1",
        "api_key": AIRNOW_API_KEY
    }

    # Define the data folder path under aqs_data_analysis
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_folder = os.path.join(root_dir, "data")
    response_folder = os.path.join(data_folder, "AirNowAPI_responses")
    os.makedirs(response_folder, exist_ok=True)

    # Final output file (combined data)
    download_file_name = "AirNowAPI_combined2020-2025.csv"
    download_file = os.path.join(response_folder, download_file_name)

    # Max date to stop looping (adjust as needed)
    max_datetime = MAX_DATETIME 

    # Track cumulative rows and start time
    total_rows = 0
    start_time = time.time()

    # Simple rate limiting: 500 requests per hour
    REQUEST_LIMIT = 400 
    TIME_WINDOW = 3600 
    request_count = 0
    hour_start_time = None

    logging.info("Starting data download process.")

    # Loop through date ranges
    while datetime.strptime(options["start_date"] + "T" + options["start_hour_utc"], "%Y-%m-%dT%H") < datetime.strptime(max_datetime, "%Y-%m-%dT%H"):
        current_time = time.time()

        # Check and enforce rate limit
        if request_count >= REQUEST_LIMIT:
            if hour_start_time is not None:
                elapsed = current_time - hour_start_time
                if elapsed < TIME_WINDOW:
                    wait_time = TIME_WINDOW - elapsed
                    logging.warning(f"Rate limit reached ({REQUEST_LIMIT} requests/hour). Waiting {wait_time:.2f} seconds.")
                    time.sleep(wait_time)
            # Reset for new hour
            request_count = 0
            hour_start_time = current_time

        logging.info(f"Starting request for date range: {options['start_date']}T{options['start_hour_utc']} to {options['end_date']}T{options['end_hour_utc']}.")
        REQUEST_URL = build_request_url(options)
        request_start = time.time()

        try:
            # Perform the API request
            response = requests.get(REQUEST_URL)
            response.raise_for_status()

            # Read CSV into pandas DataFrame
            df = pd.read_csv(pd.io.common.StringIO(response.text))
            rows_fetched = len(df)

            # Check if file exists to handle headers
            file_exists = os.path.exists(download_file)
            if file_exists:
                # Append without headers
                df.to_csv(download_file, mode='a', header=False, index=False)
            else:
                # Write with headers
                df.to_csv(download_file, index=False)

            # Update cumulative rows
            total_rows += rows_fetched
            request_duration = time.time() - request_start

            # Update rate limiting counters
            if hour_start_time is None:
                hour_start_time = current_time
            request_count += 1

            logging.info(f"Successfully fetched and appended {rows_fetched} rows for {options['start_date']} to {options['end_date']} (duration: {request_duration:.2f}s). Cumulative rows: {total_rows}.")

            # Step to the next date range
            options, at_max = step_request_date_range(options, max_datetime)

            # Rate limiting: wait 1 second between requests to avoid API limits
            time.sleep(1)

        except requests.exceptions.RequestException as e:
            # Include response details if available
            error_details = f"HTTP {response.status_code if 'response' in locals() else 'N/A'} - {e}"
            if 'response' in locals():
                error_details += f" - Response: {response.text[:500]}"  
            logging.error(f"Request failed for {options['start_date']} to {options['end_date']}: {error_details}")
            break  
        except Exception as e:
            logging.error(f"Unexpected error for {options['start_date']} to {options['end_date']}: {e}")
            sys.exit(1)

    # Final summary
    total_duration = time.time() - start_time
    logging.info(f"Download process completed. Total rows appended: {total_rows}. Total time: {total_duration:.2f}s. File: {download_file}")

if __name__ == "__main__":
    main()

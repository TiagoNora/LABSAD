import os
import pandas as pd
import requests
from pymongo import MongoClient
import warnings
from datetime import datetime
import pytz

warnings.filterwarnings('ignore')

def insert_stocks():
    # uri = 'mongodb://mongoadmin:a79c987b4dce244e9bc21620@vsgate-s1.dei.isep.ipp.pt:10777'  # MongoDB connection URI
    # db_name = 'labsad'  # Replace with your database name
    # stock_prices_collection_name = 'stockPrices'
    # stocks_collection_name = 'stocks'

    # # Initialize MongoDB client
    # client = MongoClient(uri)
    # db = client[db_name]
    # stock_prices_collection = db[stock_prices_collection_name]
    # stocks_collection = db[stocks_collection_name]

    # # API endpoint to fetch stock tickers
    # api_url = "http://localhost:8000/tickers/"

    # # Fetch list of all stocks and their names from the API
    # response = requests.get(api_url)
    # if response.status_code == 200:
    #     ticker_data = response.json()
    # else:
    #     print("Error fetching ticker data:", response.status_code)
    #     exit()

    # # Convert API data to a dictionary for quick lookup
    # ticker_dict = {item['Symbol']: item for item in ticker_data}

    # # Insert static stock details into `stocks` collection
    # for symbol, details in ticker_dict.items():
    #     stock_data = {
    #         "_id": details['Symbol'],
    #         "#": details['#'],
    #         "Company": details['Company'],
    #         "Weight": details['Weight'],
    #         "Image": details['Image']
    #     }
        
    #     # Insert stock details if not already present in the `stocks` collection
    #     stocks_collection.update_one(
    #         {"_id": stock_data["_id"]},  # Query by Symbol
    #         {"$setOnInsert": stock_data},  # Only insert if not exists
    #         upsert=True
    #     )

    # Get the directory of the current script file
    csv_directory = os.path.dirname(os.path.abspath(__file__))

    #Process each CSV file containing daily stock data
    for fname in os.listdir(csv_directory):
        if fname.endswith("_daily_data.csv"):

            print(f"processing {fname}")
            symbol = fname.split("_")[0].upper()

            csv_path = os.path.join(csv_directory, fname)
            df = pd.read_csv(csv_path)

            df['Date'] = df['Date'].apply(convert_to_utc)

            # # Step 1: Convert the 'Date' column to pandas datetime (handling timezone offsets)
            df['Date'] = pd.to_datetime(df['Date'])

           

            # # Step 2: Convert the 'Date' to UTC (removing timezone information and converting to UTC)
            # df['Date'] = df['Date'].dt.tz_convert('UTC')

            # # Step 3: Format the 'Date' column to the desired ISO 8601 format
            # df['Date'] = df['Date'].dt.strftime('%Y-%m-%dT%H:%M:%S.%fZ')  # Format to ISO 8601 with milliseconds
            # df['Date'] = df['Date'].str[:-3]  # Remove extra microseconds, keeping only 3 decimal places for milliseconds
           # df['Date'] = df['Date'].dt.tz_localize(None)
            #print(df.info())
            #print(df['Date'].head())
            
            #df['Date'] = df['Date'].dt.tz_localize(None)
            #df['Date'] = pd.to_datetime(df['Date'], errors='coerce').dt.tz_localize('America/New_York', ambiguous='NaT', nonexistent='NaT')

            # Convert to UTC for MongoDB storage
           # df['Date'] = df['Date'].dt.tz_convert('UTC')

           

            print(df['Date'].head())
            
            df.to_csv(f"{fname}", index=False)
            break
            # print(f"Processing Stock with symbol {symbol}")

            # # Get the stock details from the ticker dictionary
            # stock_details = ticker_dict.get(symbol)

            # if stock_details:
            #     csv_path = os.path.join(csv_directory, fname)
            #     df = pd.read_csv(csv_path)

            #     # Convert 'Date' column to datetime objects
            #     df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

            #     # Add 'Symbol' column for reference in stockPrices collection
            #     df['Symbol'] = stock_details['Symbol']


                
            #     # Convert DataFrame rows to dictionaries
            #     records = df.to_dict(orient='records')

            #     # Insert each row as a document into the `stockPrices` collection
            #     stock_prices_collection.insert_many(records)
                
            #     print(f"Processed and inserted time-series data for {symbol} into `stockPrices` collection.")
            # else:
            #     print(f"Symbol {symbol} not found in ticker data.")

    print("All files processed.")
    return None


def convert_to_utc(date_str):
    # Parse the original datetime string with timezone info
    date_obj = datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S%z')
    
    # Convert to UTC
    utc_date_obj = date_obj.astimezone(pytz.utc)
    
    # Format to ISO 8601 with milliseconds
    formatted_date = utc_date_obj.strftime('%Y-%m-%dT%H:%M:%S.000+00:00')
    
    return formatted_date

# Execute the function
if __name__ == "__main__":
    insert_stocks()
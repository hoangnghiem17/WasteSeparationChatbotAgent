import pandas as pd
import sqlite3
import requests

def add_lat_long_with_custom_geocoding(df, address_columns):
    """
    Adds latitude and longitude columns to a DataFrame by geocoding addresses composed of multiple columns.

    :param df: DataFrame containing the address data.
    :param address_columns: List of column names to combine into a single address string.
    :return: DataFrame with added latitude and longitude columns.
    """
    geocode_url = "https://nominatim.openstreetmap.org/search"
    headers = {
        'User-Agent': 'WasteSeparationChatbot/1.0 (nghhoang@gmail.com)'
    }

    latitudes = []
    longitudes = []

    for _, row in df.iterrows():
        # Combine the address components into a single string
        address = ", ".join(str(row[col]) for col in address_columns if pd.notna(row[col]))
        
        try:
            geocode_params = {
                'q': address,
                'format': 'json',
                'limit': 1
            }
            response = requests.get(geocode_url, headers=headers, params=geocode_params)
            response.raise_for_status()
            data = response.json()
            if data:
                latitudes.append(float(data[0]['lat']))
                longitudes.append(float(data[0]['lon']))
            else:
                latitudes.append(None)
                longitudes.append(None)
        except Exception as e:
            print(f"Error geocoding address '{address}': {e}")
            latitudes.append(None)
            longitudes.append(None)

    df['lat'] = latitudes
    df['long'] = longitudes
    return df

def excel_to_sqlite_with_query(excel_file, sqlite_db, table_name, address_columns):
    """
    Reads an Excel file, adds latitude and longitude columns based on custom Nominatim API geocoding,
    imports its data into an SQLite database table, and executes a SELECT * query to display the contents.

    :param excel_file: Path to the Excel file.
    :param sqlite_db: Path to the SQLite database file.
    :param table_name: Name of the table to store the data.
    :param address_columns: List of column names to combine into a single address string for geocoding.
    """
    try:
        # Read the Excel file into a DataFrame
        df = pd.read_excel(excel_file)

        # Add latitude and longitude columns using geocoding
        df = add_lat_long_with_custom_geocoding(df, address_columns)

        # Connect to the SQLite database (it will be created if it doesn't exist)
        conn = sqlite3.connect(sqlite_db)

        # Write the data to the specified table
        df.to_sql(table_name, conn, if_exists='replace', index=False)

        print(f"Data from '{excel_file}' has been successfully written to the '{table_name}' table in '{sqlite_db}'.")

        # Execute a SELECT * query to display the data
        query = f"SELECT * FROM {table_name}"
        result_df = pd.read_sql_query(query, conn)

        print("\nData retrieved from the database:")
        print(result_df)

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        conn.close()

# Example usage
if __name__ == "__main__":
    excel_file_path = "setup/recyclinghof.xlsx"  # Adjusted path to uploaded file
    sqlite_db_path = "waste_separation_frankfurt.db"
    table_name = "recyclinghof"
    address_columns = ["street", "zip", "district"]  # Replace with actual column names

    excel_to_sqlite_with_query(excel_file_path, sqlite_db_path, table_name, address_columns)

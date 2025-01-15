import pandas as pd
import sqlite3

def excel_to_sqlite(excel_file, sqlite_db, table_name):
    """
    Reads an Excel file and imports its data into an SQLite database table.

    :param excel_file: Path to the Excel file.
    :param sqlite_db: Path to the SQLite database file.
    :param table_name: Name of the table to store the data.
    """
    try:
        # Read the Excel file into a DataFrame
        df = pd.read_excel(excel_file)

        # Connect to the SQLite database (it will be created if it doesn't exist)
        conn = sqlite3.connect(sqlite_db)

        # Write the data to the specified table
        df.to_sql(table_name, conn, if_exists='replace', index=False)

        print(f"Data from '{excel_file}' has been successfully written to the '{table_name}' table in '{sqlite_db}'.")

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        conn.close()

# Example usage
if __name__ == "__main__":
    excel_file_path = "setup/recyclinghof.xlsx"  
    sqlite_db_path = "waste_separation_frankfurt.db"   
    table_name = "recyclinghof"          

    excel_to_sqlite(excel_file_path, sqlite_db_path, table_name)

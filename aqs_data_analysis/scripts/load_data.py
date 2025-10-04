"""
AQS Data Loader - Simple CSV to SQLite Converter
A basic utility to convert CSV files to SQLite for easier data exploration with SQL queries.

This is just a helper script to make the AQS data easier to work with.
The main analysis happens elsewhere - this just sets up the data for SQL access.
 
"""

import sqlite3
import csv
import os
from pathlib import Path
from datetime import datetime
import time

# Get the directory where this script is located
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_DIR = PROJECT_ROOT / 'data'
DATABASE_PATH = DATA_DIR / 'aqs_data.db'
PRIMARY_TABLE = 'primary_data'  # Default table name for all SQL queries

class DatabaseManager:
    """Simple CSV to SQLite converter - nothing fancy, just makes data queryable."""
    
    def __init__(self, db_path, data_dir):
        self.db_path = Path(db_path)
        self.data_dir = Path(data_dir)
    
    def get_database_info(self):
        if not self.db_path.exists():
            return None
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        try:
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            table_names = [row[0] for row in cursor.fetchall()]
            if not table_names:
                conn.close()
                return {'tables': [], 'db_size': 0}
            db_size = self.db_path.stat().st_size / (1024 * 1024)
            tables_info = []
            for table_name in table_names:
                cursor.execute(f'SELECT COUNT(*) FROM "{table_name}"')
                row_count = cursor.fetchone()[0]
                cursor.execute(f'PRAGMA table_info("{table_name}")')
                columns = cursor.fetchall()
                tables_info.append({
                    'name': table_name,
                    'row_count': row_count,
                    'column_count': len(columns),
                    'columns': [col[1] for col in columns[:5]]
                })
            conn.close()
            return {
                'tables': tables_info,
                'db_size': db_size,
                'total_rows': sum(t['row_count'] for t in tables_info)
            }
        except Exception as e:
            conn.close()
            raise Exception(f"Database query failed: {e}")

    def table_exists(self, table_name):
        if not self.db_path.exists():
            return False
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table_name,))
        exists = cursor.fetchone() is not None
        conn.close()
        return exists

    def drop_table(self, table_name):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        try:
            cursor.execute(f'DROP TABLE IF EXISTS "{table_name}"')
            conn.commit()
            return True
        except Exception as e:
            conn.rollback()
            raise Exception(f"Failed to drop table '{table_name}': {e}")
        finally:
            conn.close()

    def load_csv_to_db(self, csv_path, table_name):
        """Basic CSV import - just dumping data into SQLite for easier querying."""
        if not Path(csv_path).exists():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
        
        if self.table_exists(table_name):
            print(f"  Replacing table '{table_name}'")
            self.drop_table(table_name)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("PRAGMA synchronous = OFF")
        cursor.execute("PRAGMA journal_mode = MEMORY")
        
        try:
            print(f"  Reading: {Path(csv_path).name}")
            
            with open(csv_path, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                headers = next(reader)
                
                columns = [f'"{header}"' for header in headers]
                create_table_sql = f'CREATE TABLE "{table_name}" ({",".join(columns)} TEXT)'
                cursor.execute(create_table_sql)
                
                print(f"  Table created ({len(headers)} columns)")
                
                placeholders = ",".join(['?' for _ in headers])
                insert_sql = f'INSERT INTO "{table_name}" VALUES ({placeholders})'
                
                rows_inserted = 0
                batch_size = 10000
                batch_data = []
                
                print("  Loading data...")
                start_time = time.time()
                
                for row in reader:
                    batch_data.append(row)
                    rows_inserted += 1
                    
                    if len(batch_data) >= batch_size:
                        cursor.executemany(insert_sql, batch_data)
                        batch_data = []
                        
                        if rows_inserted % 100000 == 0:
                            print(f"    {rows_inserted:,} rows...")
                
                if batch_data:
                    cursor.executemany(insert_sql, batch_data)
                
                conn.commit()
                elapsed = time.time() - start_time
                print(f"  Done: {rows_inserted:,} rows in {elapsed:.1f}s")
                
                return rows_inserted
                
        except Exception as e:
            conn.rollback()
            raise Exception(f"CSV import failed: {e}")
        finally:
            conn.close()

    def sanitize_table_name(self, name):
        table_name = Path(name).stem
        table_name = ''.join(c if c.isalnum() or c == '_' else '_' for c in table_name)
        if table_name and table_name[0].isdigit():
            table_name = 'tbl_' + table_name
        return table_name or 'default_table'

    def copy_table_to_primary(self, source_table):
        """Copy any table to the primary table (primary_data) for consistent SQL querying."""
        if not self.table_exists(source_table):
            raise Exception(f"Source table '{source_table}' does not exist")
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Drop primary table if it exists
            cursor.execute(f'DROP TABLE IF EXISTS "{PRIMARY_TABLE}"')
            
            # Copy table structure and data
            cursor.execute(f'CREATE TABLE "{PRIMARY_TABLE}" AS SELECT * FROM "{source_table}"')
            
            # Get row count
            cursor.execute(f'SELECT COUNT(*) FROM "{PRIMARY_TABLE}"')
            row_count = cursor.fetchone()[0]
            
            conn.commit()
            return row_count
            
        except Exception as e:
            conn.rollback()
            raise Exception(f"Failed to copy table '{source_table}' to primary: {e}")
        finally:
            conn.close()

def print_separator():
    print("-" * 60)

def display_database_status():
    db_manager = DatabaseManager(DATABASE_PATH, DATA_DIR)
    
    print("\nData Overview")
    print_separator()
    
    db_info = db_manager.get_database_info()
    
    if db_info is None:
        print("No database yet")
        return
    
    if not db_info['tables']:
        print("Empty database")
        return
    
    print(f"Database: {db_info['db_size']:.1f} MB")
    print(f"Total Records: {db_info['total_rows']:,}")
    print(f"Tables: {len(db_info['tables'])}")
    
    # Highlight primary table if it exists
    primary_exists = any(table['name'] == PRIMARY_TABLE for table in db_info['tables'])
    if primary_exists:
        print(f"★ Primary table '{PRIMARY_TABLE}' is ready for SQL queries")
    else:
        print(f"⚠ No primary table yet - copy a table to '{PRIMARY_TABLE}' for easy SQL access")
    
    print("\nTables:")
    for table in db_info['tables']:
        prefix = "★ " if table['name'] == PRIMARY_TABLE else "  "
        print(f"{prefix}{table['name']}")
        print(f"    {table['row_count']:,} records, {table['column_count']} columns")
        if table['columns']:
            columns_str = ', '.join(table['columns'])
            if len(table['columns']) == 5:
                columns_str += '...'
            print(f"    Sample: {columns_str}")

def get_csv_files():
    csv_files = []
    if Path(DATA_DIR).exists():
        csv_files = [f for f in Path(DATA_DIR).glob('*.csv')]
    return sorted(csv_files)

def select_csv_file():
    csv_files = get_csv_files()
    
    if not csv_files:
        print("No CSV files found in data directory")
        print("Put CSV files in: data/")
        return None
    
    print("\nCSV Files")
    print_separator()
    
    for i, csv_file in enumerate(csv_files, 1):
        file_size = csv_file.stat().st_size / (1024 * 1024)
        mod_time = datetime.fromtimestamp(csv_file.stat().st_mtime)
        print(f"{i:2d}. {csv_file.name}")
        print(f"     {file_size:.1f} MB, {mod_time.strftime('%Y-%m-%d %H:%M')}")
    
    while True:
        try:
            choice = input(f"\nSelect file (1-{len(csv_files)}) or 'q' to quit: ").strip()
            
            if choice.lower() == 'q':
                return None
            
            index = int(choice) - 1
            if 0 <= index < len(csv_files):
                selected_file = csv_files[index]
                print(f"Selected: {selected_file.name}")
                return selected_file
            else:
                print(f"Please enter a number between 1 and {len(csv_files)}")
                
        except ValueError:
            print("Please enter a valid number")

def get_table_name(csv_path):
    db_manager = DatabaseManager(DATABASE_PATH, DATA_DIR)
    
    print("\nTable Name")
    print_separator()
    
    suggested_name = db_manager.sanitize_table_name(Path(csv_path).name)
    print(f"Suggested: {suggested_name}")
    
    user_input = input("Table name (or Enter for suggested): ").strip()
    
    if not user_input:
        return suggested_name
    
    sanitized = db_manager.sanitize_table_name(user_input)
    
    if sanitized != user_input:
        print(f"Name sanitized to: {sanitized}")
        confirm = input(f"Use sanitized name '{sanitized}'? (y/n): ").strip().lower()
        if confirm in ['y', 'yes']:
            return sanitized
        else:
            return get_table_name(csv_path)
    else:
        return sanitized

def delete_table():
    db_manager = DatabaseManager(DATABASE_PATH, DATA_DIR)
    db_info = db_manager.get_database_info()
    
    if not db_info or not db_info['tables']:
        print("No tables available for deletion")
        return
    
    print("\nTable Deletion")
    print_separator()
    
    tables = db_info['tables']
    
    print("Tables:")
    for i, table in enumerate(tables, 1):
        print(f"{i:2d}. {table['name']} ({table['row_count']:,} records)")
    
    while True:
        try:
            choice = input(f"\nDelete table (1-{len(tables)}) or 'q' to cancel: ").strip()
            
            if choice.lower() == 'q':
                return
            
            index = int(choice) - 1
            if 0 <= index < len(tables):
                table_name = tables[index]['name']
                row_count = tables[index]['row_count']
                
                confirm = input(f"\nConfirm deletion of '{table_name}' ({row_count:,} records)? (yes/no): ").strip().lower()
                
                if confirm in ['yes', 'y']:
                    try:
                        db_manager.drop_table(table_name)
                        print(f"Deleted '{table_name}'")
                    except Exception as e:
                        print(f"Delete failed: {e}")
                else:
                    print("Cancelled")
                return
            else:
                print(f"Please enter a number between 1 and {len(tables)}")
                
        except ValueError:
            print("Please enter a valid number")

def batch_import():
    csv_files = get_csv_files()
    
    if not csv_files:
        print("No CSV files for batch import")
        return
    
    print(f"\nBatch Import: {len(csv_files)} files")
    print_separator()
    
    db_manager = DatabaseManager(DATABASE_PATH, DATA_DIR)
    
    for csv_file in csv_files:
        print(f"\n{csv_file.name}")
        table_name = db_manager.sanitize_table_name(csv_file.name)
        
        try:
            rows_imported = db_manager.load_csv_to_db(csv_file, table_name)
            print(f"Success: {rows_imported:,} rows imported to '{table_name}'")
        except Exception as e:
            print(f"Error: {e}")
    
    print("\nBatch import done")

def copy_to_primary():
    """Copy a selected table to the primary table for consistent SQL access."""
    db_manager = DatabaseManager(DATABASE_PATH, DATA_DIR)
    db_info = db_manager.get_database_info()
    
    if not db_info or not db_info['tables']:
        print("No tables available to copy")
        return
    
    # Filter out primary table from options if it exists
    available_tables = [t for t in db_info['tables'] if t['name'] != PRIMARY_TABLE]
    
    if not available_tables:
        print(f"Only '{PRIMARY_TABLE}' exists - no other tables to copy")
        return
    
    print(f"\nCopy Table to Primary ('{PRIMARY_TABLE}')")
    print("This sets up the default table for all your SQL queries")
    print_separator()
    
    print("Available tables:")
    for i, table in enumerate(available_tables, 1):
        print(f"{i:2d}. {table['name']} ({table['row_count']:,} records)")
    
    while True:
        try:
            choice = input(f"\nCopy table (1-{len(available_tables)}) or 'q' to cancel: ").strip()
            
            if choice.lower() == 'q':
                return
            
            index = int(choice) - 1
            if 0 <= index < len(available_tables):
                source_table = available_tables[index]['name']
                row_count = available_tables[index]['row_count']
                
                print(f"\nCopying '{source_table}' ({row_count:,} records) to '{PRIMARY_TABLE}'...")
                
                try:
                    copied_rows = db_manager.copy_table_to_primary(source_table)
                    print(f"Success! '{PRIMARY_TABLE}' now has {copied_rows:,} records")
                    print(f"You can now use '{PRIMARY_TABLE}' in all your SQL queries")
                except Exception as e:
                    print(f"Copy failed: {e}")
                return
            else:
                print(f"Please enter a number between 1 and {len(available_tables)}")
                
        except ValueError:
            print("Please enter a valid number")

def main():
    print("AQS Data Loader")
    print("Simple CSV to SQLite converter for easier data querying")
    print(f"★ Primary table: '{PRIMARY_TABLE}' - use this name in all your SQL queries")
    print_separator()
    
    # Ensure data directory exists
    DATA_DIR.mkdir(exist_ok=True)
    
    while True:
        print("\nOptions:")
        print("1. Load single CSV file")
        print("2. Load all CSV files")
        print("3. View data overview")
        print("4. Copy table to primary (for SQL queries)")
        print("5. Delete table")
        print("6. Quit")
        
        choice = input("\nOption (1-6): ").strip()
        
        if choice == '1':
            csv_file = select_csv_file()
            if csv_file:
                table_name = get_table_name(csv_file)
                if table_name:
                    try:
                        print(f"\nLoading {csv_file.name} as '{table_name}'...")
                        db_manager = DatabaseManager(DATABASE_PATH, DATA_DIR)
                        rows_imported = db_manager.load_csv_to_db(csv_file, table_name)
                        print(f"Loaded: {rows_imported:,} rows")
                    except Exception as e:
                        print(f"Failed: {e}")
        
        elif choice == '2':
            batch_import()
        
        elif choice == '3':
            display_database_status()
        
        elif choice == '4':
            copy_to_primary()
        
        elif choice == '5':
            delete_table()
        
        elif choice == '6':
            print("Exiting...")
            break
        
        else:
            print("Invalid option. Please select 1-6.")

if __name__ == "__main__":
    main()

"""
Utility script to view data in Snowflake tables
"""
import os
import sys
import pandas as pd
import snowflake.connector
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def get_snowflake_connection():
    """Get connection to Snowflake database"""
    try:
        # Get connection parameters from environment variables
        user = os.getenv("SNOWFLAKE_USER")
        password = os.getenv("SNOWFLAKE_PASSWORD")
        account = os.getenv("SNOWFLAKE_ACCOUNT")
        warehouse = os.getenv("SNOWFLAKE_WAREHOUSE", "COMPUTE_WH")
        database = os.getenv("SNOWFLAKE_DATABASE", "HEALTHCARE_INDUSTRY_CUSTOMER_DATA")
        role = os.getenv("SNOWFLAKE_ROLE", "ACCOUNTADMIN")
        
        if not all([user, password, account]):
            print("ERROR: Snowflake connection parameters not set.")
            print("Please run setup_snowflake.py first.")
            return None
        
        # Connect to Snowflake
        conn = snowflake.connector.connect(
            user=user,
            password=password,
            account=account,
            warehouse=warehouse,
            database=database,
            role=role
        )
        
        print(f"Connected to Snowflake account: {account}")
        print(f"Database: {database}, Warehouse: {warehouse}, Role: {role}")
        
        return conn
    except Exception as e:
        print(f"ERROR: Failed to connect to Snowflake: {str(e)}")
        return None

def list_schemas(conn):
    """List all schemas in the database"""
    try:
        cursor = conn.cursor()
        cursor.execute("SHOW SCHEMAS")
        schemas = cursor.fetchall()
        cursor.close()
        
        print("\n=== Available Schemas ===")
        for i, schema in enumerate(schemas, 1):
            print(f"{i}. {schema[1]}")
        
        return [schema[1] for schema in schemas]
    except Exception as e:
        print(f"ERROR: Failed to list schemas: {str(e)}")
        return []

def list_tables(conn, schema):
    """List all tables in the specified schema"""
    try:
        cursor = conn.cursor()
        cursor.execute(f"SHOW TABLES IN SCHEMA {schema}")
        tables = cursor.fetchall()
        cursor.close()
        
        print(f"\n=== Tables in {schema} ===")
        for i, table in enumerate(tables, 1):
            print(f"{i}. {table[1]}")
        
        return [table[1] for table in tables]
    except Exception as e:
        print(f"ERROR: Failed to list tables in schema {schema}: {str(e)}")
        return []

def view_table(conn, schema, table):
    """View contents of the specified table"""
    try:
        # Query the table
        query = f"SELECT * FROM {schema}.{table} LIMIT 100"
        df = pd.read_sql(query, conn)
        
        # Print table info
        print(f"\n=== Contents of {schema}.{table} ===")
        print(f"Total rows: {len(df)}")
        print(f"Columns: {', '.join(df.columns)}\n")
        
        # Display the data
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        print(df)
        
        return df
    except Exception as e:
        print(f"ERROR: Failed to view table {schema}.{table}: {str(e)}")
        return None

def interactive_mode():
    """Interactive mode for viewing Snowflake tables"""
    conn = get_snowflake_connection()
    if not conn:
        return
    
    try:
        while True:
            # List schemas
            schemas = list_schemas(conn)
            if not schemas:
                break
            
            # Select schema
            schema_idx = input("\nEnter schema number (or 'q' to quit): ")
            if schema_idx.lower() == 'q':
                break
            
            try:
                schema = schemas[int(schema_idx) - 1]
            except (IndexError, ValueError):
                print("Invalid schema number.")
                continue
            
            # List tables in selected schema
            tables = list_tables(conn, schema)
            if not tables:
                continue
            
            # Select table
            table_idx = input(f"\nEnter table number to view (or 'b' to go back, 'q' to quit): ")
            if table_idx.lower() == 'q':
                break
            elif table_idx.lower() == 'b':
                continue
            
            try:
                table = tables[int(table_idx) - 1]
            except (IndexError, ValueError):
                print("Invalid table number.")
                continue
            
            # View selected table
            view_table(conn, schema, table)
            
            # Continue or quit
            action = input("\nPress Enter to continue, 'q' to quit: ")
            if action.lower() == 'q':
                break
    finally:
        if conn:
            conn.close()
            print("\nSnowflake connection closed.")

def specific_table(schema, table):
    """View a specific table directly"""
    conn = get_snowflake_connection()
    if not conn:
        return
    
    try:
        view_table(conn, schema, table)
    finally:
        if conn:
            conn.close()
            print("\nSnowflake connection closed.")

if __name__ == "__main__":
    if len(sys.argv) == 3:
        # If schema and table are provided as arguments
        schema = sys.argv[1]
        table = sys.argv[2]
        specific_table(schema, table)
    else:
        # Interactive mode
        interactive_mode()

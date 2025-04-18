"""
Setup script to configure Snowflake connection parameters
"""
import os
import sys
import getpass

def setup_snowflake_env():
    """
    Interactive setup for Snowflake connection parameters
    Creates a .env file with Snowflake credentials
    """
    print("\n=== MarketScope AI - Snowflake Connection Setup ===\n")
    print("Important: Make sure you have permissions to CREATE DATABASE or specify an existing database.\n")
    
    # Get Snowflake connection parameters
    account = input("Enter Snowflake account identifier (e.g., xy12345.us-east-1): ")
    user = input("Enter Snowflake username: ")
    password = getpass.getpass("Enter Snowflake password: ")
    warehouse = input("Enter Snowflake warehouse name [COMPUTE_WH]: ") or "COMPUTE_WH"
    database = input("Enter Snowflake database name [HEALTHCARE_INDUSTRY_CUSTOMER_DATA]: ") or "HEALTHCARE_INDUSTRY_CUSTOMER_DATA"
    role = input("Enter Snowflake role [ACCOUNTADMIN]: ") or "ACCOUNTADMIN"
    
    # Create .env file content
    env_content = f"""# Snowflake connection parameters
SNOWFLAKE_ACCOUNT={account}
SNOWFLAKE_USER={user}
SNOWFLAKE_PASSWORD={password}
SNOWFLAKE_WAREHOUSE={warehouse}
SNOWFLAKE_DATABASE={database}
SNOWFLAKE_ROLE={role}
"""
    
    # Check if .env file exists
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if os.path.exists(env_path):
        # Read existing .env file
        with open(env_path, "r") as f:
            existing_content = f.read()
        
        # Remove existing Snowflake parameters
        lines = existing_content.splitlines()
        filtered_lines = [line for line in lines if not line.startswith("SNOWFLAKE_")]
        
        # Add new Snowflake parameters
        new_content = "\n".join(filtered_lines) + "\n\n" + env_content
        
        # Write updated .env file
        with open(env_path, "w") as f:
            f.write(new_content)
    else:
        # Create new .env file
        with open(env_path, "w") as f:
            f.write(env_content)
    
    print("\nSnowflake connection parameters saved to .env file.")
    print("To test the connection, restart the application and upload data.")

if __name__ == "__main__":
    setup_snowflake_env()

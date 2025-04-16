"""
MarketScope AI System Runner
Orchestrates all components of the MarketScope system
"""
import subprocess
import time
import os
import signal
import sys
from typing import List
from config.config import Config

# Global process list
processes = []

def start_process(command: List[str], name: str):
    """Start a process and add it to the global processes list."""
    print(f"Starting {name}...")
    try:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        processes.append((process, name))
        print(f"{name} started with PID {process.pid}")
        return process
    except Exception as e:
        print(f"Error starting {name}: {e}")
        return None

def monitor_processes():
    """Monitor running processes and print their output."""
    try:
        while processes:
            for process, name in processes:
                # Check if process has terminated
                if process.poll() is not None:
                    print(f"{name} exited with code {process.returncode}")
                    processes.remove((process, name))
                    continue
                
                # Read output
                stdout_line = process.stdout.readline()
                if stdout_line:
                    print(f"[{name}] {stdout_line.strip()}")
                
                stderr_line = process.stderr.readline()
                if stderr_line:
                    print(f"[{name}] ERROR: {stderr_line.strip()}")
            
            # Sleep to avoid high CPU usage
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        print("Interrupted by user. Shutting down...")
        cleanup()

def cleanup():
    """Terminate all running processes."""
    for process, name in processes:
        print(f"Terminating {name}...")
        try:
            process.terminate()
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            print(f"{name} did not terminate gracefully, killing...")
            process.kill()
        except Exception as e:
            print(f"Error terminating {name}: {e}")

def validate_config():
    """Validate configuration before starting services."""
    missing = Config.validate_config()
    if missing:
        print(f"ERROR: Missing required configuration variables: {', '.join(missing)}")
        print("Please add these variables to your .env file")
        return False
    return True

def main():
    """Start all components of the MarketScope system."""
    # Validate configuration
    if not validate_config():
        return
    
    # Start the MCP server
    mcp_server = start_process(
        ["python", "mcp_server.py"],
        "MCP Server"
    )
    if not mcp_server:
        print("Failed to start MCP Server, aborting...")
        return
    
    # Wait for MCP server to initialize
    print("Waiting for MCP server to initialize...")
    time.sleep(3)
    
    # Start the FastAPI server
    api_server = start_process(
        ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", str(Config.API_PORT), "--reload"],
        "FastAPI Server"
    )
    if not api_server:
        print("Failed to start FastAPI Server, aborting...")
        cleanup()
        return
    
    # Start Streamlit frontend
    streamlit = start_process(
        ["streamlit", "run", "frontend/app.py"],
        "Streamlit Frontend"
    )
    if not streamlit:
        print("Failed to start Streamlit Frontend, aborting...")
        cleanup()
        return
    
    # Print URLs for all services
    print("\n" + "="*50)
    print("MarketScope AI is now running!")
    print("=" * 50)
    print(f"MCP Server:      {Config.MCP_URL}")
    print(f"FastAPI:         http://localhost:{Config.API_PORT}")
    print(f"API Docs:        http://localhost:{Config.API_PORT}/docs")
    print("Streamlit UI:    http://localhost:8501")
    print("=" * 50)
    print("Press Ctrl+C to shutdown all components")
    
    # Monitor processes
    monitor_processes()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nShutting down MarketScope AI...")
        cleanup()

"""
Main entry point for MarketScope platform
Starts all services: MCP servers, API, and frontend
Ensures proper initialization order and handles dependencies
"""
import os
import subprocess
import signal
import sys
import time
import atexit
import logging
import argparse
from typing import Dict, List, Optional

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("run")

# Add project root to sys.path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    logger.info(f"Added {project_root} to Python path")

# Try to import Config
try:
    from config.config import Config
    logger.info("Successfully imported Config")
except ImportError:
    logger.warning("Could not import Config, will use default values")
    # Define a minimal Config class
    class Config:
        API_PORT = int(os.getenv("API_PORT", 8001))
        MCP_PORT = int(os.getenv("MCP_PORT", 8000))

# Dictionary to track running processes
processes = {}

def check_dependencies() -> bool:
    """Check if all required dependencies are installed using Poetry"""
    try:
        # Check if Poetry is installed
        subprocess.check_call(["poetry", "--version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        # Check if dependencies are installed
        result = subprocess.run(["poetry", "check"], capture_output=True, text=True)
        if "All set!" in result.stdout:
            return True
            
        # Try to install dependencies
        print("Installing dependencies with Poetry...")
        subprocess.check_call(["poetry", "install"])
        return True
    except (subprocess.SubprocessError, FileNotFoundError):
        # Poetry not installed or error during installation
        logger.error("Poetry not installed or error during dependency installation")
        print("\n❌ Error with Poetry dependencies")
        
        # Fallback to manual import check
        required_packages = [
            "fastapi", "uvicorn", "streamlit", "langchain", "langgraph", 
            "pandas", "openai", "mcp", "pydantic", "python-dotenv"
        ]
        
        missing = []
        for package in required_packages:
            try:
                __import__(package)
            except ImportError:
                missing.append(package)
        
        if missing:
            print(f"\n❌ Missing dependencies: {', '.join(missing)}")
            print("Please run 'poetry install' manually to install all dependencies.")
            return False
        
        return True

def start_service(name: str, command: List[str], cwd: Optional[str] = None, verify_url: Optional[str] = None) -> Optional[subprocess.Popen]:
    """
    Start a service as a separate process
    
    Args:
        name: Service name
        command: Command to run
        cwd: Working directory
        
    Returns:
        Process object or None if failed
    """
    try:
        print(f"Starting {name}...")
        
        # Prepend poetry run if needed
        if command[0] == sys.executable:
            command = ["poetry", "run", "python"] + command[1:]
        elif command[0] == "streamlit":
            command = ["poetry", "run", "streamlit"] + command[1:]
        
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=cwd
        )
        
        # Wait a moment to make sure it started successfully
        time.sleep(5)  # Increased wait time for startup
        
        if process.poll() is None:
            # Check if we need to verify the service by URL
            if verify_url:
                import time
                import requests
                for attempt in range(5):  # Try 5 times
                    try:
                        print(f"Verifying {name} at {verify_url}...")
                        response = requests.get(verify_url, timeout=3)
                        if response.status_code == 200:
                            print(f"[OK] {name} started and verified at {verify_url} (PID {process.pid})")
                            return process
                        else:
                            print(f"[WARNING] {name} is running but returned status code {response.status_code}")
                    except requests.RequestException as e:
                        print(f"[WARNING] Attempt {attempt+1}/5: {name} is running but not responding to requests: {str(e)}")
                    
                    # Sleep between attempts
                    time.sleep(3)
                
                # If verification failed
                print(f"[ERROR] {name} is running but could not be verified at {verify_url}")
                return process  # Still return the process as it might start working later
            else:
                print(f"[OK] {name} started (PID {process.pid})")
                return process
        else:
            print(f"[ERROR] Failed to start {name}")
            stderr = process.stderr.read() if process.stderr else "No error output"
            print(f"Last stderr from {name}:\n{stderr}")
            return None
            
    except Exception as e:
        print(f"❌ Error starting {name}: {str(e)}")
        return None

def stop_all_services():
    """Stop all running services"""
    print("\nStopping all services...")
    
    for name, process in processes.items():
        if process and process.poll() is None:
            print(f"Stopping {name}...")
            try:
                process.terminate()
                # Give it a moment to terminate gracefully
                time.sleep(1)
                if process.poll() is None:
                    # Force kill if still running
                    process.kill()
            except Exception as e:
                print(f"Error stopping {name}: {str(e)}")
    
    print("All services stopped.")

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Run MarketScope platform")
    parser.add_argument("--no-mcp", action="store_true", help="Don't start MCP servers")
    parser.add_argument("--no-api", action="store_true", help="Don't start API server")
    parser.add_argument("--no-frontend", action="store_true", help="Don't start Streamlit frontend")
    
    args = parser.parse_args()
    
    # Check dependencies
    if not check_dependencies():
        return
    
    # Register cleanup function
    atexit.register(stop_all_services)
    
    # Define service configurations
    api_port = Config.API_PORT if 'Config' in globals() else int(os.getenv('API_PORT', 8001))
    services = [
        {
            "name": "MCP Servers",
            "command": [sys.executable, os.path.join("mcp_servers", "run_all_servers.py")],
            "skip": args.no_mcp,
            "verify_url": None  # No direct verification for MCP servers
        },
        {
            "name": "API Server",
            "command": [sys.executable, os.path.join("api", "main.py")],
            "skip": args.no_api,
            "verify_url": f"http://localhost:{api_port}/health"  # Verify API health endpoint
        },
        {
            "name": "Streamlit Frontend",
            "command": ["streamlit", "run", os.path.join("frontend", "app.py"), "--server.port", "8501"],
            "skip": args.no_frontend,
            "verify_url": "http://localhost:8501/_stcore/health"  # Verify Streamlit health
        }
    ]
    
    # Start each service in order (MCP servers first, then API, then frontend)
    for service in services:
        if not service["skip"]:
            process = start_service(
                service["name"],
                service["command"],
                verify_url=service.get("verify_url")
            )
            if process:
                processes[service["name"]] = process
                # Give servers time to start up
                time.sleep(3)
    
    # Keep the main process running and monitor services
    print("\n[SUCCESS] All services started! Access the app at http://localhost:8501")
    print("Press Ctrl+C to stop all services")
    
    try:
        # Monitor servers and restart them if they crash
        while True:
            # Check each service
            for name, process in list(processes.items()):
                if process and process.poll() is not None:
                    # Service crashed or exited
                    return_code = process.poll()
                    print(f"[WARNING] {name} exited with code {return_code}. Restarting...")
                    
                    # Get stderr for debugging
                    stderr = process.stderr.read() if process.stderr else "No error output"
                    print(f"Last stderr from {name}:\n{stderr}")
                    
                    # Find the command for this service
                    service_config = next((s for s in services if s["name"] == name), None)
                    if service_config:
                        # Restart the service
                        new_process = start_service(name, service_config["command"])
                        if new_process:
                            processes[name] = new_process
            
            # Sleep to avoid high CPU usage
            time.sleep(5)
    except KeyboardInterrupt:
        print("\nReceived interrupt signal. Shutting down...")
        sys.exit(0)

if __name__ == "__main__":
    main()

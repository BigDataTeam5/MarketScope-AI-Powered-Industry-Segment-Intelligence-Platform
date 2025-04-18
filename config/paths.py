"""
Path management for MarketScope platform
Ensures consistent path resolution across all components
"""
import os
import sys
from pathlib import Path

# Get the project root directory
PROJECT_ROOT = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def setup_paths():
    """Add project root to Python path if not already present"""
    project_root_str = str(PROJECT_ROOT)
    if project_root_str not in sys.path:
        sys.path.insert(0, project_root_str)
        print(f"Added {project_root_str} to Python path")
    
    # Also ensure the config directory is in the path
    config_dir = os.path.dirname(os.path.abspath(__file__))
    if config_dir not in sys.path:
        sys.path.insert(0, config_dir)

def get_project_path(relative_path: str) -> Path:
    """Get a path relative to the project root"""
    return PROJECT_ROOT / relative_path

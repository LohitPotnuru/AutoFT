"""Configuration Loader Utility

This module provides a reusable function to safely load YAML configuration files
with proper error handling and validation.
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional


def load_yaml_config(config_file: str, required: bool = True) -> Optional[Dict[str, Any]]:
    """Safely load a YAML configuration file.
    
    This function handles:
    - File existence checking
    - YAML parsing with error handling
    - Empty file validation
    - Path resolution (handles both absolute and relative paths)
    
    Args:
        config_file: Path to the YAML configuration file (can be absolute or relative)
        required: If True, raises exceptions for missing files. If False, returns None.
        
    Returns:
        Dictionary containing the loaded configuration, or None if file doesn't exist
        and required=False
        
    Raises:
        FileNotFoundError: If the config file doesn't exist and required=True
        yaml.YAMLError: If the YAML file is malformed
        ValueError: If the YAML file is empty or contains invalid data
    """
    config_path = Path(config_file)
    
    # Resolve the path (handles relative paths)
    if not config_path.is_absolute():
        # If relative, resolve from current working directory
        config_path = Path.cwd() / config_path
    
    # Check if file exists
    if not config_path.exists():
        if required:
            raise FileNotFoundError(
                f"Configuration file not found: {config_file} "
                f"(resolved to: {config_path.absolute()})"
            )
        return None
    
    # Check if it's a file (not a directory)
    if not config_path.is_file():
        raise ValueError(f"Path exists but is not a file: {config_path.absolute()}")
    
    # Load and parse YAML
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise yaml.YAMLError(
            f"Error parsing YAML file {config_file}: {str(e)}"
        ) from e
    except Exception as e:
        raise IOError(
            f"Error reading configuration file {config_file}: {str(e)}"
        ) from e
    
    # Validate that config is not None (empty file)
    if config is None:
        if required:
            raise ValueError(
                f"Configuration file is empty or contains no valid YAML: {config_file}"
            )
        return None
    
    # Validate that config is a dictionary
    if not isinstance(config, dict):
        raise ValueError(
            f"Configuration file must contain a YAML dictionary/mapping, "
            f"got {type(config).__name__}: {config_file}"
        )
    
    return config

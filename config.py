"""
Module Docstring:

This module defines Pydantic models for application and database configurations,
as well as a function for loading configuration data from a YAML file.

Classes:
    - AppConfig: Defines settings for the application.
    - DatabaseConfig: Defines settings for the database connection.
    - AppConfigSettings: Combines AppConfig and DatabaseConfig into a single settings model.
    - CustomAppConfigSettings: Subclass of AppConfigSettings with updated default values.

Functions:
    - load_config_from_yaml: Loads configuration data from a YAML file and merges it with CLI arguments.

Dependencies:
    - yaml: PyYAML library for YAML parsing.
    - pydantic.BaseModel: Base class for Pydantic models.
    - pydantic.Field: Field type for defining model fields.

"""

from typing import Optional

import yaml
from pydantic import BaseModel, Field


class AppConfig(BaseModel):
    app_name: str = Field(..., description="Name of the application")
    debug: bool = Field(False, description="Enable debug mode")
    log_level: str = Field("info", description="Logging level")


class DatabaseConfig(BaseModel):
    url: str = Field(..., description="Database connection URL")
    max_connections: Optional[int] = Field(
        None, description="Maximum number of database connections"
    )
    timeout: int = Field(30, description="Database connection timeout (seconds)")


class InferenceConfig(BaseModel):
    model_path: str = Field(..., description="Path to the model")
    tokenizer_path: str = Field(..., description="Path to the tokenizer")
    device_map: Optional[str] = Field(None, description="Device mapping")
    max_memory: Optional[str] = Field(None, description="Maximum memory")


class ApiConfig(BaseModel):
    host: str = Field(..., description="API host address")
    port: int = Field(..., description="API port number")


class AppConfigSettings(BaseModel):
    app: AppConfig = Field(..., description="Application settings")
    db: DatabaseConfig = Field(..., description="Database settings")
    inference: InferenceConfig = Field(..., description="Inference settings")
    api: ApiConfig = Field(..., description="API settings")

    class Config:
        env_prefix = "APP_"  # Prefix for environment variables
        case_sensitive = False  # Case sensitivity for environment variables


# Load configuration data from a YAML file
def load_config_from_yaml(file_path: str, cli_args: dict) -> AppConfigSettings:
    """Load configuration data from a YAML file and merge it with CLI
    arguments.

    Args:
        file_path (str): Path to the YAML configuration file.
        cli_args (dict): Dictionary containing CLI arguments.

    Returns:
        AppConfigSettings: Merged configuration settings.

    Raises:
        FileNotFoundError: If the specified YAML file does not exist.
        yaml.YAMLError: If there is an error parsing the YAML file.
    """
    with open(file_path, "r", encoding="utf-8") as yaml_file:
        config_data = yaml.safe_load(yaml_file)

    # Merge configuration data with CLI arguments
    for key, value in cli_args.items():
        for subkey in ["app", "db", "inference", "api"]:
            if key in config_data.get(subkey, {}):
                config_data[subkey][key] = value

    return AppConfigSettings(**config_data)

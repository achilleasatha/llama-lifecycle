"""Entrypoint module."""

from config import load_config_from_yaml


def main(config=None):
    """Main function of your application."""
    if config is None:
        config = load_config_from_yaml(file_path="app_config.yaml", cli_args={})

    app_name = config.app.app_name
    debug = config.app.debug
    log_level = config.app.log_level

    # Your application logic using the configuration settings
    print(f"Application Name: {app_name}")
    print(f'Debug Mode: {"Enabled" if debug else "Disabled"}')
    print(f"Log Level: {log_level}")


if __name__ == "__main__":
    main()

import json

import click

from config import load_config_from_yaml
from llama_lifecycle.main import main


@click.command()
@click.option(
    "--config-file", default="app_config.yaml", help="Path to configuration file"
)
@click.option("--app-name", default="MyApp", help="Name of the application")
@click.option("--debug/--no-debug", default=False, help="Enable debug mode")
@click.option("--log-level", default="info", help="Logging level")
def run_cli(config_file, app_name, debug, log_level):  # pylint:disable=unused-argument
    """Command-line interface for the application."""
    cli_args = {k: v for k, v in locals().items() if v is not None}
    config = load_config_from_yaml(config_file, cli_args)
    click.echo(json.dumps(config.app.dict(), indent=2))  # pylint:disable=no-member

    # Invoke the function from main.py
    main(config=config)


# This allows the module to be executed as a script
if __name__ == "__main__":
    run_cli()  # pylint:disable=no-value-for-parameter

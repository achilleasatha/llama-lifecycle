# Configuration Loader

This code file provides an entry point module for an application that loads configuration settings from a YAML file using the `load_config_from_yaml` function from the `config` module.

## Main Function

The `main` function of the application is responsible for loading the configuration settings and then using those settings in the application logic. If no configuration is provided as an argument to the function, it will load the configuration from the "app_config.yaml" file by default.

Within the `main` function:
- The application name, debug mode, and log level are extracted from the configuration settings.
- The application name is printed.
- The debug mode is printed as either "Enabled" or "Disabled" depending on the setting.
- The log level is printed.

## How to Use

To use this code file, you can simply run it as the main module by executing the script. The configuration settings will be loaded from the specified YAML file, or the default file if none is provided.

## License Information

This project is licensed under the MIT License. For more details, please refer to the [LICENSE](https://github.com/achilleasatha/llama_lifecycle/blob/main/LICENSE) file.

## Version Information

You can find the version information for this project tagged on GitHub. For more details, refer to the [GitHub Releases](https://github.com/achilleasatha/llama-lifecycle/releases) page.

## Continuous Integration

This project utilizes GitHub Actions for continuous integration. You can check the status of the workflows in the [GitHub Actions](https://github.com/achilleasatha/llama-lifecycle/actions/workflows/ci.yml) page.

## Code Coverage

Code coverage information for this project is available through Codecov. You can view the code coverage report [here](https://codecov.io/gh/achilleasatha/llama-lifecycle).

Feel free to explore the project setup, development steps, and contribution guidelines provided in this README for more information on setting up and working with the project. If you encounter any issues or have questions, please open an issue on GitHub for support.
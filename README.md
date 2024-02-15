# Application Configuration

This project provides an example of how to manage an LLM deployment lifecycle for a real-world application using Python. The code in the `main.py` file is the entry point for the application and loads configuration settings from a YAML file using the `config` module.

## Running the Application

To run the application, execute the `main` function in the `main.py` file. If no configuration is provided, the application will load the settings from the file `app_config.yaml`.

The `main` function then retrieves the application name, debug mode, and log level from the configuration settings and prints them to the console.

Example output:
- Application Name: My App
- Debug Mode: Enabled
- Log Level: INFO

## Project Setup and Development

To set up and develop this project locally, follow the steps outlined below:

### Prerequisites

Make sure you have the following tools installed on your system:
- Python (version 3.12 or higher)
- Poetry (Python dependency management tool)
- Git (Version control system)

### Installation

1. Clone the Repository:
   ```bash
   git clone https://github.com/your_username/your_project.git
   ```

2. Navigate to the Project Directory:
   ```bash
   cd your_project
   ```

3. Install Dependencies with Poetry:
   ```bash
   poetry install
   ```

### Development Commands

- **Run Tests:**
  ```bash
  poetry run pytest
  ```

- **Run pre-commit Hooks (Manually):**
  ```bash
  poetry run pre-commit run --all-files
  ```

- **Lint and Format Code:**
  ```bash
  poetry run black .
  poetry run isort .
  ```

### Adding Custom Configuration

You can customize the application settings by modifying the `app_config.yaml` file. Feel free to add additional configuration options as needed for your project.

For any contributions, please refer to the Contributing Guidelines in the repository. If you need support or have any questions, feel free to open an issue on GitHub. 

Enjoy working on your Llama Lifecycle project! 🚀
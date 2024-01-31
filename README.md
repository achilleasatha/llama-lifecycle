[![MIT License](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/achilleasatha/llama_lifecycle/blob/main/LICENSE)
[![GitHub tag](https://img.shields.io/github/tag/achilleasatha/llama_lifecycle.svg)](https://github.com/achilleasatha/llama-lifecycle/releases)
![GitHub Actions](https://github.com/achilleasatha/llama-lifecycle/actions/workflows/ci.yml/badge.svg)
[![codecov](https://codecov.io/gh/achilleasatha/llama-lifecycle/graph/badge.svg?token=LIYRZKK6W3)](https://codecov.io/gh/achilleasatha/llama-lifecycle)

# Llama Lifecycle

An example toy project on how to manage an LLM deployment lifecycle for a real-world application.

## Project Setup

To set up and develop this project locally, follow these steps:

### Prerequisites

Make sure you have the following tools installed on your system:

- [Python](https://www.python.org/) (version 3.12 or higher)
- [Poetry](https://python-poetry.org/) (Python dependency management tool)
- [Git](https://git-scm.com/) (Version control system)

### Installation

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/your_username/your_project.git
    ```

2. **Navigate to the Project Directory:**

    ```bash
    cd your_project
    ```

3. **Install Dependencies with Poetry:**

    ```bash
    poetry install
    ```
This command will create a virtual environment and install the project dependencies specified in pyproject.toml.

### Set up pre-commit Hooks:

```bash
poetry run pre-commit install
```
This command will set up pre-commit hooks defined in .pre-commit-config.yaml to run before each commit, ensuring code quality and consistency.

### Development
You can now start developing your project. Here are some useful commands:

**Run Tests:**

```bash
poetry run pytest
```

**Run pre-commit Hooks (Manually):**
```bash
poetry run pre-commit run --all-files
```

**Lint and Format Code:**
```bash
poetry run black .
poetry run isort .
```

### Contributing
If you'd like to contribute to this project, please follow the Contributing Guidelines.

### Support
If you encounter any issues or have questions about this project, please open an issue on GitHub.

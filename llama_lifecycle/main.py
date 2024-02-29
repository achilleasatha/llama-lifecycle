"""Entrypoint module."""

import uvicorn
from api.chatbot_api import ChatbotAPI
from inference.inference import LlamaChatbot

from config import load_config_from_yaml


def main(config=None):
    """Main function of your application."""
    if config is None:
        config = load_config_from_yaml(file_path="app_config.yaml", cli_args={})

    app_name = config.app.app_name
    debug = config.app.debug
    log_level = config.app.log_level
    print(f"Application Name: {app_name}")
    print(f'Debug Mode: {"Enabled" if debug else "Disabled"}')
    print(f"Log Level: {log_level}")

    chatbot = LlamaChatbot(
        model_path=config.inference.model_path,
        tokenizer_path=config.inference.tokenizer_path,
        device_map=config.inference.device_map,
        max_memory=config.inference.max_memory,
    )
    chatbot_api = ChatbotAPI(chatbot=chatbot)
    uvicorn.run(chatbot_api.app, host=config.api.host, port=config.api.port)


if __name__ == "__main__":
    main()

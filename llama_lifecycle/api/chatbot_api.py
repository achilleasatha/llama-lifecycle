from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from pydantic import BaseModel

from llama_lifecycle.inference.inference import LlamaChatbot


class InferenceRequest(BaseModel):
    prompt: str


class InferenceResponse(BaseModel):
    generated_text: str


class ChatbotAPI:
    def __init__(self, chatbot: LlamaChatbot):
        self.app = FastAPI()
        self.setup_routes()
        self.chatbot = chatbot

    def setup_routes(self):
        @self.app.post("/generate_text/")
        async def generate_text(request: InferenceRequest) -> InferenceResponse:
            prompt = request.prompt
            generated_text = self.chatbot.generate_response(prompt)
            return InferenceResponse(generated_text=generated_text)

        @self.app.get("/docs")
        async def get_docs():
            return RedirectResponse(url="/docs")

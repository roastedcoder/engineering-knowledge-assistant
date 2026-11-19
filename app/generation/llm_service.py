
import os
import httpx

from app.core.exceptions import RAGServiceError


class LLMService:
    def __init__(
        self,
        model: str = "qwen3.5:4b",
        base_url: str | None = None,
    ):
        self.model = model
        self.base_url = base_url or os.getenv(
            "OLLAMA_BASE_URL",
            "http://localhost:11434",
        )

    def generate(self, prompt: str, system_prompt: str = "") -> str:
        payload = {
            "model": self.model,
            "think": False,
            "messages": [
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            "stream": False,
        }

        try:
            response = httpx.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=120.0,
            )

            response.raise_for_status()

            result = response.json()
            return result["message"]["content"]

        except httpx.HTTPStatusError as error:
            raise RAGServiceError(
                f"Ollama returned HTTP {error.response.status_code}"
            ) from error

        except httpx.RequestError as error:
            raise RAGServiceError(
                f"Could not connect to Ollama: {error}"
            ) from error

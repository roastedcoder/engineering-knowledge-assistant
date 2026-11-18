
import httpx


class LLMService:

    def __init__(
        self,
        model: str = "qwen3.5:4b",
        base_url: str = "http://localhost:11434",
    ):
        self.model = model
        self.base_url = base_url

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

        response = httpx.post(
            f"{self.base_url}/api/chat",
            json=payload,
            timeout=120.0,
        )

        response.raise_for_status()

        result = response.json()

        return result["message"]["content"]

import os

from dotenv import load_dotenv
import requests

from utils.config_loader import load_config

load_dotenv()


class OpenRouterClient:

    def __init__(self):

        self.enabled = bool(os.getenv("OPENROUTER_API_KEY"))
        self.last_status = "disabled" if not self.enabled else "ready"
        self.last_error = None
        self.config = load_config("config/system_config.yaml")
        self.model = self.config["llm"].get("model", "openrouter/free")
        self.max_tokens = int(self.config["llm"].get("max_tokens", 1200))
        self.base_url = "https://openrouter.ai/api/v1/chat/completions"
        self.site_url = os.getenv("OPENROUTER_SITE_URL", "http://localhost")
        self.app_name = os.getenv("OPENROUTER_APP_NAME", "RosterIQ")

    def _extract_text(self, payload):

        choices = payload.get("choices", [])
        if not choices:
            return None

        message = choices[0].get("message", {})
        content = message.get("content")

        if isinstance(content, str):
            return content.strip() or None

        if isinstance(content, list):
            text_parts = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    text_parts.append(item.get("text", ""))
            merged = "".join(text_parts).strip()
            return merged or None

        return None

    def generate(self, prompt, system_prompt=None, max_tokens=None, temperature=None):

        if not self.enabled:
            self.last_status = "disabled"
            self.last_error = "OPENROUTER_API_KEY missing."
            return None

        headers = {
            "Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY')}",
            "Content-Type": "application/json",
            "HTTP-Referer": self.site_url,
            "X-Title": self.app_name,
        }
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        body = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.2 if temperature is None else float(temperature),
            "max_tokens": self.max_tokens if max_tokens is None else int(max_tokens),
        }

        try:
            response = requests.post(
                self.base_url,
                headers=headers,
                json=body,
                timeout=45,
            )
            response.raise_for_status()
            text = self._extract_text(response.json())
            if text:
                self.last_status = "success"
                self.last_error = None
                return text

            self.last_status = "empty_response"
            self.last_error = "OpenRouter returned an empty response."
            return None
        except requests.HTTPError as exc:
            error_text = exc.response.text if exc.response is not None else str(exc)
            self.last_status = "error"
            self.last_error = error_text
            return None
        except Exception as exc:
            self.last_status = "error"
            self.last_error = str(exc)
            return None

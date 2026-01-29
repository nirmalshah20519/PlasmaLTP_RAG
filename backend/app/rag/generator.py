"""Response generation functionality."""
from typing import List, Dict, Any
import logging
import requests
from app.core.config import settings
from app.rag.prompt import build_rag_prompt
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# Together chat completions endpoint (OpenAI-compatible)
TOGETHER_CHAT_URL = "https://api.together.xyz/v1/chat/completions"
# Fallback chat model if configured model returns 400 (e.g. deprecated name)
TOGETHER_CHAT_FALLBACK_MODEL = "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"


class Generator:
    """Handles response generation using Together AI chat completions API."""

    def __init__(self):
        """Initialize the generator with API configuration."""
        self.api_key = settings.TOGETHER_API_KEY
        self.model = settings.CHAT_MODEL
        self.api_url = TOGETHER_CHAT_URL
        logger.info(f"Initialized Generator with model: {self.model}")

    def generate(self, query: str, evidence: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate a response based on query and evidence.

        Args:
            query: User query
            evidence: Retrieved evidence with citation information

        Returns:
            Dictionary with 'answer' and 'metadata'
        """
        prompt = build_rag_prompt(query, evidence)

        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }
            payload = {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 512,
                "temperature": 0.7,
                "top_p": 0.9,
            }

            response = requests.post(
                self.api_url,
                json=payload,
                headers=headers,
                timeout=120,
            )

            if not response.ok:
                try:
                    err_body = response.json()
                    err_msg = err_body.get("error", {}).get("message", response.text[:500])
                except Exception:
                    err_msg = response.text[:500]
                logger.error(
                    "Together chat API error: status=%s, body=%s",
                    response.status_code,
                    err_msg,
                )
                # If 400 and configured model may be deprecated, retry with fallback
                if response.status_code == 400 and self.model != TOGETHER_CHAT_FALLBACK_MODEL:
                    if "model" in (err_msg or "").lower() or "invalid" in (err_msg or "").lower():
                        logger.warning(
                            "Retrying with fallback chat model: %s",
                            TOGETHER_CHAT_FALLBACK_MODEL,
                        )
                        payload["model"] = TOGETHER_CHAT_FALLBACK_MODEL
                        response = requests.post(
                            self.api_url,
                            json=payload,
                            headers=headers,
                            timeout=120,
                        )
                if not response.ok:
                    response.raise_for_status()

            data = response.json()
            choices = data.get("choices", [])
            if not choices:
                raise RuntimeError("No choices in chat completion response")

            message = choices[0].get("message", {})
            content = message.get("content", "")
            answer = (content or "").strip()

            usage = data.get("usage", {}) or {}
            total_tokens = usage.get("total_tokens")

            return {
                "answer": answer,
                "metadata": {
                    "model": data.get("model", self.model),
                    "evidence_count": len(evidence),
                    "tokens_used": total_tokens,
                }
            }

        except requests.exceptions.RequestException as e:
            logger.exception("Failed to generate response")
            raise RuntimeError(f"Failed to generate response: {e}")

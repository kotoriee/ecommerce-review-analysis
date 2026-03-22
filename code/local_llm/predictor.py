"""
Ollama Predictor - High-level Inference API

Uses direct HTTP requests to Ollama API (no external client package needed).
"""

import sys
import time
import json
import logging
import requests
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

# Allow running as script or importing as module
_CODE_DIR = Path(__file__).parent.parent
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

from local_llm.schemas import LLMPrediction, BatchPredictionResult, create_prediction_from_output
from local_llm.prompt_templates import format_chat_messages, get_few_shot_examples

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class OllamaConfig:
    base_url: str = "http://localhost:11434"
    model_name: str = "qwen3-4b-sentiment-lora"
    timeout: int = 30
    temperature: float = 0.1
    max_tokens: int = 256


def _build_few_shot_messages(text: str, language: str, n_examples: int = 3) -> List[dict]:
    """Build chat messages with few-shot examples prepended."""
    examples = get_few_shot_examples(language, n_examples)
    messages = []
    for ex in examples:
        messages.append({"role": "user", "content": ex["user"]})
        messages.append({"role": "assistant", "content": ex["assistant"]})
    # Add actual query
    base_messages = format_chat_messages(text, language)
    messages.extend(base_messages[1:])  # skip system prompt duplication
    messages.insert(0, base_messages[0])  # put system prompt first
    return messages


class OllamaClient:
    def __init__(self, config: OllamaConfig):
        self.config = config

    def chat(self, messages: List[dict], json_mode: bool = False) -> dict:
        payload = {
            "model": self.config.model_name,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": self.config.temperature,
                "num_predict": self.config.max_tokens,
            },
        }
        if json_mode:
            payload["format"] = "json"

        resp = requests.post(
            f"{self.config.base_url}/api/chat",
            json=payload,
            timeout=self.config.timeout,
        )
        resp.raise_for_status()
        return resp.json()

    def health_check(self) -> bool:
        try:
            resp = requests.get(f"{self.config.base_url}/api/tags", timeout=5)
            return resp.status_code == 200
        except Exception:
            return False

    def model_exists(self) -> bool:
        try:
            resp = requests.get(f"{self.config.base_url}/api/tags", timeout=5)
            if resp.status_code != 200:
                return False
            models = [m["name"] for m in resp.json().get("models", [])]
            return any(self.config.model_name in m for m in models)
        except Exception:
            return False


class OllamaPredictor:
    def __init__(self, config: Optional[OllamaConfig] = None, model_name: Optional[str] = None,
                 use_few_shot: bool = True, n_few_shot_examples: int = 3):
        if config is None:
            config = OllamaConfig()
        if model_name:
            config = OllamaConfig(
                base_url=config.base_url, model_name=model_name, timeout=config.timeout,
                temperature=config.temperature, max_tokens=config.max_tokens,
            )
        self.config = config
        self.client = OllamaClient(config)
        self.use_few_shot = use_few_shot
        self.n_few_shot_examples = n_few_shot_examples
        logger.info(f"OllamaPredictor initialized with model: {config.model_name}")

    def predict(self, text: str, language: str, max_retries: int = 2) -> LLMPrediction:
        if language not in ("zh", "en", "ru"):
            raise ValueError(f"Unsupported language: {language}")

        if self.use_few_shot:
            messages = _build_few_shot_messages(text, language, self.n_few_shot_examples)
        else:
            messages = format_chat_messages(text, language)

        last_error = None
        for attempt in range(max_retries + 1):
            try:
                return self._predict_with_messages(messages)
            except Exception as e:
                last_error = e
                logger.warning(f"Attempt {attempt + 1} failed: {e}")
                if attempt < max_retries:
                    time.sleep(0.5 * (attempt + 1))

        raise RuntimeError(f"All {max_retries + 1} attempts failed: {last_error}")

    def _predict_with_messages(self, messages: List[dict]) -> LLMPrediction:
        start_time = time.time()
        response = self.client.chat(messages=messages, json_mode=True)
        latency_ms = (time.time() - start_time) * 1000

        raw_output = None
        if "message" in response:
            message = response["message"]
            if message.get("content"):
                raw_output = message["content"]

        if not raw_output:
            raise ValueError(f"Empty response: {response}")

        return create_prediction_from_output(raw_output=raw_output, model_name=self.config.model_name, latency_ms=latency_ms)

    def predict_batch(self, texts: List[str], language: str, batch_size: int = 8, show_progress: bool = True) -> BatchPredictionResult:
        predictions = []
        for i, text in enumerate(texts):
            try:
                predictions.append(self.predict(text, language))
                if show_progress and (i + 1) % batch_size == 0:
                    logger.info(f"Processed {i + 1}/{len(texts)}")
            except Exception as e:
                logger.error(f"Failed: {e}")
                predictions.append(LLMPrediction(
                    sentiment_label=1, confidence=0.0,
                    rationale=f"Error: {e}", raw_output=None,
                    model_name=self.config.model_name, latency_ms=0.0
                ))

        result = BatchPredictionResult(predictions=predictions, total_count=len(predictions))
        result.compute_statistics()
        return result

    def health_check(self) -> bool:
        return self.client.health_check() and self.client.model_exists()


def quick_predict(text: str, language: str = "en", model_name: str = "qwen3-4b-sentiment-lora") -> LLMPrediction:
    return OllamaPredictor(model_name=model_name).predict(text, language)

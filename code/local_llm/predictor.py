"""
Ollama Predictor - High-level Inference API
"""

import time
import logging
from typing import List, Optional

from local_llm.client.ollama_client import OllamaClient, create_client
from local_llm.client.config import OllamaConfig
from local_llm.utils.prompt_templates import format_with_few_shot, format_chat_messages
from local_llm.inference.schemas import LLMPrediction, BatchPredictionResult, create_prediction_from_output

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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
        
        messages = format_with_few_shot(text, language, self.n_few_shot_examples) if self.use_few_shot else format_chat_messages(text, language)
        
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
            elif message.get("thinking"):
                raw_output = self._extract_json_from_thinking(message["thinking"])
        
        if not raw_output:
            raise ValueError(f"Empty response: {response}")
        
        return create_prediction_from_output(raw_output=raw_output, model_name=self.config.model_name, latency_ms=latency_ms)

    def _extract_json_from_thinking(self, thinking: str) -> str:
        """Extract JSON from thinking field."""
        import re
        import json
        
        # Try to find JSON object
        patterns = [
            r'\{[^{}]*"sentiment"\s*:\s*[0-2][^{}]*\}',
            r'\{[\s\S]*?"sentiment"\s*:\s*[0-2][\s\S]*?\}',
        ]
        
        for pattern in patterns:
            for match in re.findall(pattern, thinking):
                try:
                    parsed = json.loads(match)
                    if 'sentiment' in parsed:
                        return json.dumps(parsed, ensure_ascii=False)
                except:
                    continue
        
        # Extract fields
        s = re.search(r'"sentiment"\s*:\s*([0-2])', thinking)
        if s:
            c = re.search(r'"confidence"\s*:\s*([0-9.]+)', thinking)
            r = re.search(r'"rationale"\s*:\s*"([^"]+)"', thinking)
            return json.dumps({
                "sentiment": int(s.group(1)),
                "confidence": float(c.group(1)) if c else 0.8,
                "rationale": r.group(1) if r else "From reasoning"
            }, ensure_ascii=False)
        
        return thinking

    def predict_batch(self, texts: List[str], language: str, batch_size: int = 8, show_progress: bool = True) -> BatchPredictionResult:
        predictions = []
        for i, text in enumerate(texts):
            try:
                predictions.append(self.predict(text, language))
                if show_progress and (i + 1) % batch_size == 0:
                    logger.info(f"Processed {i + 1}/{len(texts)}")
            except Exception as e:
                logger.error(f"Failed: {e}")
                predictions.append(LLMPrediction(sentiment_label=1, confidence=0.0, rationale=f"Error: {e}",
                                                  raw_output=None, model_name=self.config.model_name, latency_ms=0.0))
        
        result = BatchPredictionResult(predictions=predictions, total_count=len(predictions))
        result.compute_statistics()
        return result

    def health_check(self) -> bool:
        return self.client.health_check() and self.client.model_exists()


def quick_predict(text: str, language: str = "en", model_name: str = "qwen3.5:4b") -> LLMPrediction:
    return OllamaPredictor(model_name=model_name).predict(text, language)

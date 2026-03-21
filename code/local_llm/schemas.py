"""
Output Schemas for Local LLM Inference

Defines Pydantic models for LLM prediction results with validation.
"""

import json
import re
from typing import Optional, Literal, Any
from pydantic import BaseModel, Field, field_validator


class LLMPrediction(BaseModel):
    """
    Schema for LLM sentiment prediction output.

    This model represents the structured output from the local LLM
    for sentiment classification tasks.

    Attributes:
        sentiment_label: Predicted sentiment (0=negative, 1=neutral, 2=positive)
        confidence: Confidence score between 0.0 and 1.0
        rationale: Brief reasoning explanation for the prediction
        raw_output: Raw LLM output text (for debugging)
        model_name: Name of the model used for prediction
        latency_ms: Inference latency in milliseconds
    """

    sentiment_label: Literal[0, 1, 2] = Field(
        ...,
        description="Predicted sentiment: 0=negative, 1=neutral, 2=positive"
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence score between 0.0 and 1.0"
    )
    rationale: Optional[str] = Field(
        default=None,
        description="Brief reasoning explanation for the prediction"
    )
    raw_output: Optional[str] = Field(
        default=None,
        description="Raw LLM output text for debugging"
    )
    model_name: Optional[str] = Field(
        default=None,
        description="Name of the model used for prediction"
    )
    latency_ms: Optional[float] = Field(
        default=None,
        ge=0,
        description="Inference latency in milliseconds"
    )

    @field_validator('confidence')
    @classmethod
    def round_confidence(cls, v: float) -> float:
        """Round confidence to 4 decimal places for readability."""
        return round(v, 4)

    @field_validator('rationale')
    @classmethod
    def truncate_rationale(cls, v: Optional[str]) -> Optional[str]:
        """Truncate rationale to max 500 characters."""
        if v and len(v) > 500:
            return v[:497] + "..."
        return v

    class Config:
        json_schema_extra = {
            "example": {
                "sentiment_label": 2,
                "confidence": 0.95,
                "rationale": "User uses positive words like 'amazing', 'great quality'",
                "raw_output": '{"sentiment": 2, "confidence": 0.95, "rationale": "..."}',
                "model_name": "qwen3.5:4b",
                "latency_ms": 245.5
            }
        }


class BatchPredictionResult(BaseModel):
    """
    Schema for batch prediction results.

    Contains multiple LLMPrediction results with aggregate statistics.
    """

    predictions: list[LLMPrediction] = Field(
        default_factory=list,
        description="List of individual predictions"
    )
    total_count: int = Field(
        ...,
        ge=0,
        description="Total number of predictions"
    )
    avg_confidence: Optional[float] = Field(
        default=None,
        description="Average confidence across all predictions"
    )
    avg_latency_ms: Optional[float] = Field(
        default=None,
        description="Average latency in milliseconds"
    )
    model_name: Optional[str] = Field(
        default=None,
        description="Model name used for predictions"
    )

    @field_validator('predictions')
    @classmethod
    def validate_predictions(cls, v: list[LLMPrediction]) -> list[LLMPrediction]:
        """Ensure predictions list is not empty if total_count > 0."""
        return v

    def compute_statistics(self) -> "BatchPredictionResult":
        """Compute aggregate statistics from predictions."""
        if not self.predictions:
            return self

        confidences = [p.confidence for p in self.predictions]
        latencies = [p.latency_ms for p in self.predictions if p.latency_ms is not None]

        self.avg_confidence = round(sum(confidences) / len(confidences), 4)
        if latencies:
            self.avg_latency_ms = round(sum(latencies) / len(latencies), 2)

        # Set model name from first prediction if not set
        if not self.model_name and self.predictions[0].model_name:
            self.model_name = self.predictions[0].model_name

        return self


def parse_llm_output(raw_output: str) -> dict[str, Any]:
    """
    Parse raw LLM output to extract JSON structure.

    Handles various output formats including:
    - Pure JSON: {"sentiment": 2, ...}
    - JSON with markdown: ```json\n{...}\n```
    - JSON with extra text

    Args:
        raw_output: Raw text output from LLM

    Returns:
        Dict with parsed fields (sentiment, confidence, rationale)

    Raises:
        ValueError: If output cannot be parsed
    """
    # Try to extract JSON from markdown code blocks
    json_match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', raw_output, re.DOTALL)
    if json_match:
        json_str = json_match.group(1)
    else:
        # Try to find JSON object directly
        json_match = re.search(r'\{.*\}', raw_output, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
        else:
            json_str = raw_output

    # Clean up the string
    json_str = json_str.strip()

    try:
        parsed = json.loads(json_str)
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse LLM output as JSON: {e}") from e

    # Normalize field names (handle various naming conventions)
    result = {}

    # sentiment_label / sentiment
    if "sentiment" in parsed:
        result["sentiment_label"] = parsed["sentiment"]
    elif "sentiment_label" in parsed:
        result["sentiment_label"] = parsed["sentiment_label"]
    elif "label" in parsed:
        result["sentiment_label"] = parsed["label"]
    else:
        raise ValueError("Missing required field: sentiment/sentiment_label/label")

    # confidence (optional, default to 0.5)
    result["confidence"] = parsed.get("confidence", 0.5)

    # rationale (optional)
    result["rationale"] = parsed.get("rationale", parsed.get("reason", parsed.get("explanation")))

    return result


def create_prediction_from_output(
    raw_output: str,
    model_name: str,
    latency_ms: float
) -> LLMPrediction:
    """
    Create LLMPrediction from raw LLM output.

    Args:
        raw_output: Raw text from LLM
        model_name: Name of the model
        latency_ms: Inference latency

    Returns:
        Validated LLMPrediction object

    Raises:
        ValueError: If output cannot be parsed or validated
    """
    parsed = parse_llm_output(raw_output)

    return LLMPrediction(
        sentiment_label=parsed["sentiment_label"],
        confidence=parsed.get("confidence", 0.5),
        rationale=parsed.get("rationale"),
        raw_output=raw_output,
        model_name=model_name,
        latency_ms=round(latency_ms, 2)
    )

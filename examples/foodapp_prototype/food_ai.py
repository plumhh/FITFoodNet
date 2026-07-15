from __future__ import annotations

import logging
import os
import re
from typing import Any


LOGGER = logging.getLogger(__name__)

MODEL_NAME = "deepseek-chat"
TEMPERATURE = 0.2
TOP_P = 1.0
MAX_TOKENS = 512
PROMPT_PROTOCOL_VERSION = "foodlens-label-conditioned-v1"

FOOD_SYSTEM_PROMPT = """
You are FoodLens, an English-only assistant in a research prototype for
label-conditioned food information generation.

Input limitations:
- You do not receive or inspect the original image.
- You receive only the classifier's predicted labels, confidence scores, and
  the user's question.
- Treat the top predicted label as a fallible model output, not verified fact.

Response rules:
- Answer in concise, polished English using plain text only.
- Base the answer only on the supplied labels, confidence scores, and question.
- State uncertainty when the classifier confidence or candidate labels do not
  support a confident answer.
- Provide only general food information and nutrition-oriented educational
  reference.
- Do not claim exact calories, nutrient quantities, portion size, ingredients,
  preparation method, or dietary compliance unless a verified external source
  is explicitly supplied.
- Do not determine hidden allergens, cross-contamination, gluten-free status,
  clinical suitability, or whether a food is safe for a specific person.
- For allergy or food-safety questions, explicitly state that visual
  recognition cannot verify hidden ingredients or cross-contamination and
  recommend checking ingredient records, packaging, or the food provider.
- For medical, disease-specific, pregnancy, or child-feeding questions, avoid
  clinical recommendations and advise consulting an appropriately qualified
  professional when individualized guidance is needed.
- Do not use markdown, bullet syntax, or headings.
""".strip()


class FoodAIError(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


class FoodAIService:
    def __init__(self, *, evaluation_mode: bool = True) -> None:
        self.api_key = os.getenv("DEEPSEEK_API_KEY", "").strip()
        self.base_url = (
            os.getenv("DEEPSEEK_BASE_URL", "").strip()
            or "https://api.deepseek.com"
        )
        self.model = MODEL_NAME
        self.temperature = TEMPERATURE
        self.top_p = TOP_P
        self.max_tokens = MAX_TOKENS
        self.evaluation_mode = evaluation_mode

    @property
    def configured(self) -> bool:
        return bool(self.api_key)

    def protocol_payload(self) -> dict[str, Any]:
        return {
            "provider": "deepseek",
            "model": self.model,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
            "evaluation_mode": self.evaluation_mode,
            "fallback_enabled": not self.evaluation_mode,
            "prompt_protocol": PROMPT_PROTOCOL_VERSION,
            "llm_receives_original_image": False,
        }

    def ask(
        self,
        *,
        message: str,
        dish_name: str | None,
        predictions: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        message = (message or "").strip() or (
            "Provide concise general food information for the predicted label."
        )
        dish_name = (dish_name or "").strip() or "Unknown dish"
        predictions = predictions or []

        if not self.configured:
            if self.evaluation_mode:
                raise FoodAIError(
                    "missing_api_key",
                    "DeepSeek evaluation failed because DEEPSEEK_API_KEY is not configured.",
                )
            return self._fallback_result(
                message=message,
                dish_name=dish_name,
                predictions=predictions,
                reason="missing_api_key",
            )

        try:
            answer = self._ask_openai(
                message=message,
                dish_name=dish_name,
                predictions=predictions,
            )
            if not answer:
                raise RuntimeError("DeepSeek returned an empty response.")

            return {
                "status": "success",
                "answer": self._normalize_answer(answer),
                "mode": "evaluation" if self.evaluation_mode else "live",
                "dish_name": dish_name,
                **self.protocol_payload(),
            }
        except Exception as exc:
            LOGGER.exception(
                "FoodLens LLM request failed",
                extra={
                    "provider": "deepseek",
                    "model": self.model,
                    "evaluation_mode": self.evaluation_mode,
                    "prompt_protocol": PROMPT_PROTOCOL_VERSION,
                },
            )
            if self.evaluation_mode:
                raise FoodAIError(
                    "api_request_failed",
                    "DeepSeek evaluation request failed; no fallback response was generated.",
                ) from exc
            return self._fallback_result(
                message=message,
                dish_name=dish_name,
                predictions=predictions,
                reason="api_request_failed",
            )

    def _ask_openai(
        self,
        *,
        message: str,
        dish_name: str,
        predictions: list[dict[str, Any]],
    ) -> str:
        from openai import OpenAI

        client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": FOOD_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": self._compose_user_prompt(
                        message=message,
                        dish_name=dish_name,
                        predictions=predictions,
                    ),
                },
            ],
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_tokens,
            stream=False,
        )
        choice = response.choices[0].message.content if response.choices else ""
        return (choice or "").strip()

    def _compose_user_prompt(
        self,
        *,
        message: str,
        dish_name: str,
        predictions: list[dict[str, Any]],
    ) -> str:
        lines = [
            f"Protocol: {PROMPT_PROTOCOL_VERSION}",
            "Task: label-conditioned food information generation",
            "The original image is not provided to the LLM.",
            f"Top-1 predicted label: {dish_name}",
            "Top-3 predicted labels and confidence scores:",
        ]

        if predictions:
            for item in predictions[:3]:
                label = str(item.get("label", "Unknown")).strip() or "Unknown"
                confidence = float(item.get("confidence", 0.0)) * 100
                lines.append(f"{label}: {confidence:.2f}%")
        else:
            lines.append(f"{dish_name}: confidence unavailable")

        lines.extend(
            [
                f"User question: {message}",
                (
                    "Answer using only the supplied classifier outputs and user "
                    "question. Do not imply that you inspected the image. Provide "
                    "general nutrition-oriented reference only; do not provide exact "
                    "calorie or nutrient values, hidden-allergen judgments, dietary "
                    "compliance decisions, or clinical recommendations."
                ),
            ]
        )
        return "\n".join(lines)

    def _fallback_result(
        self,
        *,
        message: str,
        dish_name: str,
        predictions: list[dict[str, Any]],
        reason: str,
    ) -> dict[str, Any]:
        return {
            "status": "fallback",
            "answer": self._safe_fallback_answer(
                message=message,
                dish_name=dish_name,
                predictions=predictions,
            ),
            "mode": "demo",
            "dish_name": dish_name,
            "failure_reason": reason,
            **self.protocol_payload(),
        }

    def _safe_fallback_answer(
        self,
        *,
        message: str,
        dish_name: str,
        predictions: list[dict[str, Any]],
    ) -> str:
        lowered = message.lower()
        uncertainty = ""
        if predictions:
            top_confidence = float(predictions[0].get("confidence", 0.0))
            if top_confidence < 0.5:
                uncertainty = " The classifier confidence is limited, so this label should be verified."

        if any(
            term in lowered
            for term in (
                "allergy",
                "allergen",
                "peanut",
                "gluten",
                "cross-contamination",
                "safe to eat",
            )
        ):
            return (
                f"The current predicted label is {dish_name}.{uncertainty} "
                "Visual recognition cannot verify hidden allergens, ingredient substitutions, "
                "or cross-contamination. Check packaging, ingredient records, or confirm with "
                "the food provider before making an allergy-related decision."
            )

        return (
            f"The current predicted label is {dish_name}.{uncertainty} "
            "This prototype can provide general food information based on that label, "
            "but it cannot verify exact ingredients, portion size, nutrient quantities, "
            "clinical suitability, or dietary safety."
        )

    def _normalize_answer(self, text: str) -> str:
        cleaned = (text or "").strip()
        if not cleaned:
            return ""

        cleaned = cleaned.replace("**", "")
        cleaned = cleaned.replace("__", "")
        cleaned = cleaned.replace("`", "")
        cleaned = re.sub(r"^\s{0,3}#{1,6}\s*", "", cleaned, flags=re.MULTILINE)
        cleaned = re.sub(r"^\s*[-*]\s+", "", cleaned, flags=re.MULTILINE)
        cleaned = re.sub(r"\[(.*?)\]\((.*?)\)", r"\1", cleaned)
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
        return cleaned.strip()

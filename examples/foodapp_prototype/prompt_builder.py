"""Prompt builder and optional LLM service for the FoodApp prototype."""

from __future__ import annotations

import os
from typing import Any


FOOD_SYSTEM_PROMPT = """
You are Food AI, an English-only assistant inside a mobile food recognition app.
Stay focused on food, dishes, ingredients, taste, nutrition, and dining context.
Use plain text only. Keep the answer concise, concrete, and suitable for a mobile screen.
If a dish label is provided, treat it as the current best visual prediction rather than a verified fact.
""".strip()


def build_user_prompt(*, message: str, dish_name: str, predictions: list[dict[str, Any]]) -> str:
    lines = [
        f"Current top dish label: {dish_name}",
        f"User request: {message}",
    ]
    if predictions:
        lines.append("Top recognition candidates:")
        for item in predictions[:3]:
            label = item.get("label", "Unknown")
            confidence = float(item.get("confidence", 0.0)) * 100
            lines.append(f"- {label}: {confidence:.2f}%")
    lines.append("Reply with a short dish description, likely ingredients, and one nutrition-oriented suggestion.")
    return "\n".join(lines)


class FoodAIService:
    def __init__(self) -> None:
        self.api_key = os.getenv("DEEPSEEK_API_KEY", "").strip() or os.getenv("OPENAI_API_KEY", "").strip()
        self.base_url = (
            os.getenv("DEEPSEEK_BASE_URL", "").strip()
            or os.getenv("OPENAI_BASE_URL", "").strip()
            or "https://api.openai.com/v1"
        )
        self.model = os.getenv("DEEPSEEK_MODEL", "").strip() or os.getenv("OPENAI_MODEL", "").strip() or "gpt-4o-mini"

    @property
    def configured(self) -> bool:
        return bool(self.api_key)

    def ask(
        self,
        *,
        message: str,
        dish_name: str | None,
        predictions: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        message = (message or "").strip() or "Describe this dish and provide one nutrition suggestion."
        dish_name = (dish_name or "").strip() or "Unknown dish"
        predictions = predictions or []

        if self.configured:
            try:
                answer = self._ask_llm(message=message, dish_name=dish_name, predictions=predictions)
                if answer:
                    return {
                        "answer": answer,
                        "mode": "live",
                        "model": self.model,
                        "dish_name": dish_name,
                    }
            except Exception:
                pass

        return {
            "answer": self._fallback_answer(message=message, dish_name=dish_name, predictions=predictions),
            "mode": "local-fallback",
            "model": None,
            "dish_name": dish_name,
        }

    def _ask_llm(self, *, message: str, dish_name: str, predictions: list[dict[str, Any]]) -> str:
        from openai import OpenAI

        client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": FOOD_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": build_user_prompt(
                        message=message,
                        dish_name=dish_name,
                        predictions=predictions,
                    ),
                },
            ],
        )
        return (response.choices[0].message.content or "").strip()

    def _fallback_answer(self, *, message: str, dish_name: str, predictions: list[dict[str, Any]]) -> str:
        alternatives = ""
        if len(predictions) > 1:
            labels = ", ".join(item.get("label", "Unknown") for item in predictions[1:3])
            alternatives = f" Alternative visual candidates include {labels}."

        lowered = message.lower()
        if "calorie" in lowered or "kcal" in lowered:
            return (
                f"{dish_name} is the current top visual prediction. A practical calorie estimate would depend on "
                f"portion size, oil, sauce, and cooking method.{alternatives}"
            )
        if "ingredient" in lowered:
            return (
                f"{dish_name} likely contains a main ingredient base, aromatics, seasoning, and a sauce or cooking "
                f"medium. Use this as an application-level hint rather than a verified ingredient list.{alternatives}"
            )
        return (
            f"{dish_name} is the strongest visual match from FITFoodNet. The app can use this label to generate a "
            f"short dietary explanation, likely ingredient notes, and a simple nutrition suggestion.{alternatives}"
        )

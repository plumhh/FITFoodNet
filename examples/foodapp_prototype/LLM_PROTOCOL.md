# FoodLens LLM Evaluation Protocol

## Positioning

FoodLens uses label-conditioned food information generation. FITFoodNet receives
the original food image. The LLM does not receive the image.

The LLM input contains only:

1. The Top-1 predicted food label.
2. The Top-3 predicted labels and confidence scores.
3. The user's question.

## Fixed inference configuration

- Provider: DeepSeek
- Model: `deepseek-chat`
- Temperature: `0.2`
- Top-p: `1.0`
- Maximum output tokens: `512`
- Prompt protocol: `foodlens-label-conditioned-v1`
- Evaluation mode: enabled
- Local fallback during evaluation: disabled

The API key is read only from `DEEPSEEK_API_KEY`. The optional base URL is read
from `DEEPSEEK_BASE_URL`; otherwise, `https://api.deepseek.com` is used.

## Evaluation failure policy

When evaluation mode is enabled, a missing API key, API exception, or empty API
response returns `status=failed`. No local template response is substituted.

## Safety boundary

Generated responses are limited to general food information and
nutrition-oriented educational reference. The system must not claim:

- Hidden-allergen or cross-contamination detection.
- Gluten-free or dietary-compliance verification.
- Clinical or disease-specific suitability.
- Exact calorie, nutrient, ingredient, or portion estimates.
- Safety for a specific individual based only on a predicted label.

For allergy and food-safety questions, the response must state that visual
recognition cannot verify hidden ingredients or cross-contamination and direct
the user to ingredient records, packaging, or the food provider.

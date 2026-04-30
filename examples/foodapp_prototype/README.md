# FoodApp Prototype

This directory provides a lightweight prototype that connects FITFoodNet to a
mobile-oriented dietary-analysis workflow:

```text
food image -> FITFoodNet prediction -> prompt builder -> LLM or local fallback -> dietary feedback
```

The prototype is included to illustrate the application scenario discussed in
the paper. It is not required for reproducing the benchmark results.

## Files

- `app.py`: Flask API and web entry point.
- `predictor.py`: FITFoodNet checkpoint loading and top-k prediction logic.
- `prompt_builder.py`: prompt construction and LLM/local fallback response logic.
- `templates/index.html`: mobile-style demo page.
- `static/app.js`: frontend interaction logic.
- `static/styles.css`: demo page styling.
- `.env.example`: environment variable template.

## Setup

Install the main repository environment first, then install the prototype-only
dependencies:

```bash
pip install -r examples/foodapp_prototype/requirements.txt
```

Copy `.env.example` to `.env` or export the same variables in your shell:

```bash
FITFOODNET_CHECKPOINT=/path/to/fitfoodnet_best.pth
FITFOODNET_CLASS_JSON=/path/to/class_indices.json
FITFOODNET_NUM_CLASSES=172
FITFOODNET_DINOV3_REPO=/path/to/dinov3
FITFOODNET_DINOV3_SOURCE=local
FITFOODNET_DINOV3_WEIGHT=/path/to/dinov3_vitl16_pretrain.pth
```

LLM access is optional. If no API key is configured, the app returns a local
fallback response:

```bash
OPENAI_API_KEY=your_api_key_here
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_MODEL=gpt-4o-mini
```

DeepSeek-compatible variables are also supported:

```bash
DEEPSEEK_API_KEY=your_api_key_here
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-chat
```

## Run

```bash
python examples/foodapp_prototype/app.py
```

Then open:

```text
http://127.0.0.1:5000
```

## Notes

- Do not commit `.env`, uploaded images, checkpoints, or API keys.
- This prototype uses fixed image-level classification. It does not implement
  portion estimation, multi-food detection, or user-study functionality.

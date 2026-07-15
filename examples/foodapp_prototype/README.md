# FoodLens Prototype

FoodLens is the mobile-oriented application prototype associated with
FITFoodNet. The browser sends a food image to the Flask server, FITFoodNet
returns the top recognition candidates, and DeepSeek generates concise food
information conditioned on those labels.

```text
food image -> FITFoodNet top-k labels -> fixed prompt protocol -> deepseek-chat
```

The LLM does not receive the original image. It receives only the top-1 label,
the top-3 labels with confidence scores, and the user's question. The fixed
configuration and safety boundary are documented in `LLM_PROTOCOL.md`.

## Files

- `app.py`: Flask entry point and JSON API.
- `predictor.py`: FITFoodNet checkpoint loading and top-k inference.
- `food_ai.py`: fixed DeepSeek configuration, prompts, and failure handling.
- `LLM_PROTOCOL.md`: reproducible LLM protocol and safety boundary.
- `templates/index.html`: mobile-oriented FoodLens interface.
- `static/app.js`: camera/upload and conversation workflow.
- `static/styles.css`: responsive mobile styling.

## Configuration

Install the repository environment and the prototype dependencies:

```bash
pip install -r requirements.txt
pip install -r examples/foodapp_prototype/requirements.txt
```

Copy `.env.example` to `.env`, then set local paths and the DeepSeek API key.
Checkpoints, DINOv3 weights, datasets, uploaded images, and API keys must not be
committed to the repository.

The class JSON file may be either a list of class names or an index-to-name
mapping. `FITFOODNET_NUM_CLASSES` must match the checkpoint classifier.

## Run

From the repository root:

```bash
python examples/foodapp_prototype/app.py
```

Open `http://127.0.0.1:5000` on the same computer. For a phone on the same
local network, open `http://<computer-lan-ip>:5000`.

## Scope

FoodLens is a research prototype for fine-grained food recognition and
label-conditioned food information generation. It does not verify hidden
allergens, cross-contamination, exact nutrient quantities, portion size,
dietary compliance, or clinical suitability.

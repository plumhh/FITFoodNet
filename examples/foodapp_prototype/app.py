"""Flask prototype for FITFoodNet-based dietary feedback."""

from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request

from predictor import ModelSetupError, PredictionError, PredictorService
from prompt_builder import FoodAIService


BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parents[1]
load_dotenv(BASE_DIR / ".env")

app = Flask(__name__)
predictor = PredictorService(repo_root=REPO_ROOT)
food_ai = FoodAIService()


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/api/health")
def health():
    return jsonify(
        {
            "ok": True,
            "food_ai_configured": food_ai.configured,
            **predictor.status_payload(),
        }
    )


@app.post("/api/predict")
def predict():
    try:
        image_file = request.files.get("image")
        if image_file is None or not image_file.filename:
            return (
                jsonify(
                    {
                        "ok": False,
                        "error_type": "bad_request",
                        "message": "Upload an image file with form field 'image'.",
                    }
                ),
                400,
            )

        result = predictor.predict_bytes(
            image_file.read(),
            filename=image_file.filename,
        )
        return jsonify({"ok": True, **result})
    except ModelSetupError as exc:
        return jsonify({"ok": False, "error_type": "model_setup", "message": str(exc)}), 503
    except PredictionError as exc:
        return jsonify({"ok": False, "error_type": "prediction_failed", "message": str(exc)}), 500


@app.post("/api/food-ai")
def ask_food_ai():
    payload = request.get_json(silent=True) or {}
    reply = food_ai.ask(
        message=payload.get("message", ""),
        dish_name=payload.get("dish_name"),
        predictions=payload.get("predictions") or [],
    )
    return jsonify({"ok": True, **reply})


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)

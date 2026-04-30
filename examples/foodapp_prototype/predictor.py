"""FITFoodNet inference service used by the FoodApp prototype."""

from __future__ import annotations

import io
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image, UnidentifiedImageError


class PredictionError(RuntimeError):
    pass


class ModelSetupError(RuntimeError):
    pass


@dataclass
class PredictorConfig:
    checkpoint: Path
    class_json: Path | None
    num_classes: int
    model_name: str
    dinov3_repo: str
    dinov3_source: str
    dinov3_weight: Path | None
    img_size: int = 224


class PredictorService:
    def __init__(self, repo_root: Path) -> None:
        self.repo_root = repo_root
        if str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))

        self.config = self._load_config()
        self._runtime: dict[str, Any] | None = None

    @property
    def model_ready(self) -> bool:
        return self._runtime is not None

    def status_payload(self) -> dict[str, Any]:
        checkpoint = self.config.checkpoint
        if not checkpoint.is_file():
            return {
                "model_ready": False,
                "checkpoint": str(checkpoint),
                "details": "Checkpoint file does not exist.",
            }
        if self._runtime is None:
            return {
                "model_ready": False,
                "checkpoint": str(checkpoint),
                "details": "Checkpoint found. Model will initialize on first prediction.",
            }
        return {
            "model_ready": True,
            "checkpoint": str(checkpoint),
            "details": "Model loaded.",
        }

    def predict_bytes(self, image_bytes: bytes, *, filename: str) -> dict[str, Any]:
        runtime = self._ensure_runtime()
        torch = runtime["torch"]
        model = runtime["model"]
        transform = runtime["transform"]
        class_names = runtime["class_names"]
        device = runtime["device"]

        try:
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        except UnidentifiedImageError as exc:
            raise PredictionError("Could not decode image. Use JPG, PNG, or WebP.") from exc

        tensor = transform(image).unsqueeze(0).to(device)
        with torch.inference_mode():
            logits = model(tensor)

        probs = torch.softmax(logits, dim=-1)
        scores, indices = torch.topk(probs, k=min(5, probs.shape[-1]))

        predictions = []
        for score, index in zip(scores[0].tolist(), indices[0].tolist()):
            predictions.append(
                {
                    "label": class_names[index] if index < len(class_names) else f"class_{index}",
                    "index": int(index),
                    "confidence": round(float(score), 6),
                }
            )

        return {
            "filename": filename,
            "top_prediction": predictions[0] if predictions else None,
            "predictions": predictions,
        }

    def _load_config(self) -> PredictorConfig:
        checkpoint = Path(os.getenv("FITFOODNET_CHECKPOINT", "")).expanduser()
        class_json_raw = os.getenv("FITFOODNET_CLASS_JSON", "").strip()
        dinov3_weight_raw = os.getenv("FITFOODNET_DINOV3_WEIGHT", "").strip()

        return PredictorConfig(
            checkpoint=checkpoint,
            class_json=Path(class_json_raw).expanduser() if class_json_raw else None,
            num_classes=int(os.getenv("FITFOODNET_NUM_CLASSES", "172")),
            model_name=os.getenv("FITFOODNET_MODEL_NAME", "dinov3_l"),
            dinov3_repo=os.getenv("FITFOODNET_DINOV3_REPO", "facebookresearch/dinov3"),
            dinov3_source=os.getenv("FITFOODNET_DINOV3_SOURCE", "github"),
            dinov3_weight=Path(dinov3_weight_raw).expanduser() if dinov3_weight_raw else None,
        )

    def _ensure_runtime(self) -> dict[str, Any]:
        if self._runtime is not None:
            return self._runtime

        if not self.config.checkpoint.is_file():
            raise ModelSetupError(f"Checkpoint not found: {self.config.checkpoint}")

        try:
            import torch
            from torchvision import transforms
            from fitfoodnet.model import FITFoodNet
        except ImportError as exc:
            raise ModelSetupError("Missing dependencies. Install the repository requirements first.") from exc

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = FITFoodNet(
            num_classes=self.config.num_classes,
            model_type=self.config.model_name,
            img_size=self.config.img_size,
            dinov3_repo=self.config.dinov3_repo,
            dinov3_source=self.config.dinov3_source,
            dinov3_weight=self.config.dinov3_weight,
        )
        state_dict = torch.load(self.config.checkpoint, map_location="cpu")
        if any(key.startswith("module.") for key in state_dict.keys()):
            state_dict = {key.replace("module.", "", 1): value for key, value in state_dict.items()}
        model.load_state_dict(state_dict, strict=True)
        model.to(device).eval()

        transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(self.config.img_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        self._runtime = {
            "torch": torch,
            "model": model,
            "transform": transform,
            "class_names": self._load_class_names(),
            "device": device,
        }
        return self._runtime

    def _load_class_names(self) -> list[str]:
        if self.config.class_json and self.config.class_json.is_file():
            data = json.loads(self.config.class_json.read_text(encoding="utf-8"))
            if isinstance(data, list):
                return [str(item) for item in data]
            if isinstance(data, dict):
                return [str(data.get(str(index), index)) for index in range(self.config.num_classes)]
        return [f"class_{index}" for index in range(self.config.num_classes)]

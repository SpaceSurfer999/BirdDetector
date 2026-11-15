"""
Модуль инференса BirdDetector.
Загружает обученную модель и делает предсказания на кадрах с камеры.
"""

from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from bird_detector.config import load_config
from bird_detector.data import get_transforms
from bird_detector.model import build_model


def imread_unicode(path: str) -> np.ndarray:
    """Читает изображение по пути с любыми символами (обход ограничения cv2.imread на Windows)."""
    with open(path, "rb") as f:
        bytes_data = f.read()
    array = np.frombuffer(bytes_data, dtype=np.uint8)
    frame = cv2.imdecode(array, cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError(f"Не удалось декодировать изображение: {path}")
    return frame


def load_model(
    model_path: str, num_classes: int = 2, device: str = "cpu"
) -> torch.nn.Module:
    """Загружает обученную модель из .pth файла."""
    model = build_model(num_classes=num_classes)
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def preprocess_frame(frame: np.ndarray, transforms) -> torch.Tensor:
    """Конвертирует кадр OpenCV (BGR, HWC) -> тензор [1, 3, 224, 224]."""
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(rgb)
    tensor = transforms(image)
    tensor = tensor.unsqueeze(0)
    return tensor


def predict(
    model: torch.nn.Module,
    frame: np.ndarray,
    transforms,
    device: str,
    threshold: float,
    classes: list[str],
) -> dict:
    """Предсказание на одном кадре с применением порога уверенности."""
    tensor = preprocess_frame(frame, transforms).to(device)

    with torch.no_grad():
        logits = model(tensor)

    probs = F.softmax(logits, dim=1)[0]  # Берем первый элемент батча (у нас размер 1)
    confidence, pred_idx = torch.max(probs, 0)

    class_name = classes[pred_idx.item()]
    is_bird = (class_name == "1_bird") and (confidence.item() > threshold)

    return {
        "class_name": class_name,
        "confidence": round(confidence.item(), 4),
        "is_bird": is_bird,
    }


if __name__ == "__main__":
    # 1. Загружаем конфигурацию
    cfg = load_config()
    inf_cfg = cfg.get("inference", {})

    if not inf_cfg:
        raise KeyError("Секция 'inference' отсутствует в configs/default.yaml")

    model_path = inf_cfg["model_path"]
    threshold = inf_cfg["threshold"]
    classes = inf_cfg["classes"]
    test_imgs = inf_cfg["test_images"]

    # 2. Выбираем устройство
    requested_device = inf_cfg.get("device", "auto")
    if requested_device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = requested_device

    print(f"[CONFIG] Device: {device}, Threshold: {threshold}")
    print(f"[CONFIG] Model path: {model_path}")

    # 3. Инициализация модели и трансформаций
    # Проверяем существование файла модели перед загрузкой
    if not Path(model_path).exists():
        raise FileNotFoundError(
            f"Файл модели не найден: {model_path}. Сначала запусти обучение!"
        )

    model = load_model(model_path, num_classes=len(classes), device=device)
    transforms = get_transforms(is_train=False)

    # 4. Тест на реальных изображениях из конфига
    for label, img_path in test_imgs.items():
        print(f"\n--- Testing '{label}' ({img_path}) ---")
        try:
            frame = imread_unicode(img_path)
            result = predict(model, frame, transforms, device, threshold, classes)

            status = "BIRD DETECTED" if result["is_bird"] else "NO BIRD"
            print(f"Result: {status}")
            print(f"Class: {result['class_name']}, Confidence: {result['confidence']}")

        except (
            OSError,
            ValueError,
        ) as e:  # Ловим конкретные ошибки файлов и декодирования
            print(f"Error processing {img_path}: {e}")

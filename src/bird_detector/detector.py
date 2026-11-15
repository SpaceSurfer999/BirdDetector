"""
Модуль живого детектора птиц.
Использует функции из infer.py для предсказания на кадрах с камеры в реальном времени.
"""

import time
from datetime import datetime
from pathlib import Path

import cv2
import torch

from bird_detector.config import load_config
from bird_detector.data import get_transforms
from bird_detector.infer import load_model, predict


def run_live_detection():
    """
    Главный цикл работы детектора.
    Запускается бесконечно, пока не будет остановлен пользователем (Ctrl+C).
    """
    # 1. Загрузка конфигурации
    cfg = load_config()

    # Берем настройки из секций 'inference' и 'detector'
    inf_cfg = cfg.get("inference", {})
    det_cfg = cfg.get("detector", {})

    if not inf_cfg or not det_cfg:
        raise KeyError("Missing 'inference' or 'detector' section in config")

    model_path = inf_cfg["model_path"]
    classes = inf_cfg["classes"]

    # Логика выбора порога: override или дефолт
    threshold = det_cfg.get("threshold_override") or inf_cfg["threshold"]

    camera_id = det_cfg.get("camera_id", 0)
    skip_frames = det_cfg.get("skip_frames", 5)
    cooldown_sec = det_cfg.get("cooldown_sec", 10)
    output_dir = Path(det_cfg.get("output_dir", "data/detections"))

    # Подготовка устройства
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"[DETECTOR] Initializing... Device: {device}, Threshold: {threshold}")
    print(
        f"[DETECTOR] Camera ID: {camera_id}, Skip frames: {skip_frames}, Cooldown: {cooldown_sec}s"
    )

    # 2. Инициализация модели и трансформаций
    # Важно: делаем это ОДИН РАЗ перед циклом, а не внутри каждого кадра!
    model = load_model(model_path, num_classes=len(classes), device=device)
    transforms = get_transforms(is_train=False)  # Импорт из data.py

    # Создание папки для результатов
    output_dir.mkdir(parents=True, exist_ok=True)

    # 3. Открытие камеры
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera with ID {camera_id}")

    print("[DETECTOR] Camera opened. Starting loop... Press Ctrl+C to stop.")

    last_capture_time = 0.0
    frame_count = 0

    try:
        while True:
            ret, frame = cap.read()

            if not ret:
                print("[WARN] Frame grab failed. Retrying...")
                continue

            frame_count += 1

            # --- ЗАДАЧА 1: THROTTLING (Ограничение частоты) ---
            # Если счетчик кадров делится на skip_frames без остатка -> обрабатываем.
            # Иначе -> пропускаем (continue).
            # Зачем: ResNet тяжелый, гонять его на 30 FPS бессмысленно.
            if frame_count % skip_frames != 0:
                continue

            current_time = time.time()

            # --- ЗАДАЧА 2: COOLDOWN (Задержка повторного срабатывания) ---
            # Проверяем, прошло ли достаточно времени с последнего снимка.
            # Если нет -> пропускаем обработку (не тратим GPU зря).
            if current_time - last_capture_time < cooldown_sec:
                continue

            # --- ЗАДАЧА 3: INFERENCE & DECISION ---
            # Используем готовую функцию predict из infer.py
            # Она сама сделает BGR->RGB, трансформации, forward pass и вернет dict
            result = predict(
                model=model,
                frame=frame,  # Сырой кадр от OpenCV (BGR)
                transforms=transforms,
                device=device,
                threshold=threshold,
                classes=classes,
            )

            # --- ЗАДАЧА 4: ACTION (Сохранение) ---
            # Если птица обнаружена уверенно (> threshold):
            if result["is_bird"]:
                timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"bird_{timestamp_str}_conf{result['confidence']:.2f}.jpg"
                filepath = output_dir / filename

                # Сохраняем оригинальный BGR кадр (cv2.imwrite ждет BGR)
                success = cv2.imwrite(str(filepath), frame)

                if success:
                    print(
                        f"[CAPTURE] Saved: {filepath.name} | Conf: {result['confidence']}"
                    )
                    # Обновляем время последнего снимка для cooldown
                    last_capture_time = current_time
                else:
                    print(f"[ERROR] Failed to save image to {filepath}")

            # Опционально: вывод статуса каждые N кадров, чтобы видеть, что скрипт жив
            elif frame_count % 100 == 0:
                print(
                    f"[STATUS] Alive. Frames processed: {frame_count}, Last capture: {(current_time - last_capture_time):.1f}s ago"
                )

    except KeyboardInterrupt:
        print("\n[INFO] Stopped by user.")
    finally:
        cap.release()
        print("[INFO] Camera released. Detector stopped.")


if __name__ == "__main__":
    run_live_detection()

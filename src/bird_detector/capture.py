"""
Модуль сбора кадров с веб-камеры.

Читает конфиг из configs/default.yaml, снимает кадры каждые N секунд,
фильтрует по движению, сохраняет с таймстампами.
"""

import logging
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import yaml


def load_config(config_path: str = "configs/default.yaml") -> dict:
    """Загружаем конфигурацию из YAML файла"""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def setup_logging(config: dict) -> logging.Logger:
    """Настраивает логирование по конфигу"""
    log_config = config["logging"]
    log_file = Path(log_config["log_file"])
    log_file.parent.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=getattr(logging, log_config["level"]),
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )
    logger = logging.getLogger(__name__)
    return logger


def detect_motion(
    prev_frame: np.ndarray, current_frame: np.ndarray, threshold: float
) -> tuple[bool, float]:
    """
    Определяет движение между двумя кадрами.

    Args:
        prev_frame: предыдущий кадр (grayscale)
        current_frame: текущий кадр (grayscale)
        threshold: порог движения (среднее отклонение)

    Returns:
        Кортеж (motion_detected, mean_diff)
    """
    diff = cv2.absdiff(prev_frame, current_frame)
    mean_diff = np.mean(diff)
    return mean_diff >= threshold, mean_diff


def capture_loop(config: dict, logger: logging.Logger):
    """Основной цикл съемки кадров"""
    capture_config = config["capture"]

    interval = capture_config["interval_seconds"]
    threshold = capture_config["motion_threshold"]
    max_frames = capture_config["max_frames"]
    save_dir = Path(capture_config["save_dir"])
    camera_id = capture_config["camera_id"]

    save_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(camera_id)
    cap.set(cv2.CAP_PROP_ZOOM, 50)
    if not cap.isOpened():
        logger.error(f"Не удалось открыть камеру {camera_id}")
        return

    logger.info(
        f"Камера открыта. Интервал: {interval} сек, порог движения: {threshold}"
    )
    logger.info(f"Сохранение в: {save_dir}")

    prev_frame = None
    frame_count = 0
    skipped_count = 0

    try:
        while frame_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                logger.error("Ошибка чтения кадра")
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # первый кадр сохраняем всегда
            if prev_frame is None:
                motion = True
                mean_diff = float("inf")
            else:
                motion, mean_diff = detect_motion(prev_frame, gray, threshold)

            # если движение есть — сохраняем кадр
            if motion:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = save_dir / f"frame_{timestamp}.jpg"
                cv2.imwrite(str(filename), frame)

                frame_count += 1
                logger.info(
                    f"[{frame_count}] Сохранён: {filename.name} (diff: {mean_diff:.2f})"
                )
            else:
                # движения нет — пропускаем сохранение
                skipped_count += 1

            # всегда обновляем prev_frame и ждём
            prev_frame = gray
            time.sleep(interval)

    except KeyboardInterrupt:
        logger.info("Остановлено пользователем")

    finally:
        cap.release()
        logger.info(f"Всего сохранено: {frame_count}, пропущено: {skipped_count}")


if __name__ == "__main__":
    config = load_config()
    logger = setup_logging(config)
    logger.info("Запуск автосбора кадров")
    capture_loop(config, logger)

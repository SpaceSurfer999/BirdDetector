"""
Модуль загрузки и доступа к конфигурации проекта.
Использует PyYAML для безопасного чтения configs/default.yaml.
"""

from pathlib import Path
from typing import Any

import yaml


def load_config(config_name: str = "default.yaml") -> dict[str, Any]:
    """
    Загружает конфигурацию из папки configs/.

    Args:
        config_name: Имя файла конфига (по умолчанию default.yaml).

    Returns:
        Словарь с данными конфигурации.
    """
    # Определяем корень проекта относительно этого файла
    # src/bird_detector/config.py -> ../../configs/default.yaml
    project_root = Path(__file__).parent.parent.parent
    config_path = project_root / "configs" / config_name

    if not config_path.exists():
        raise FileNotFoundError(f"Конфиг не найден: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        # safe_load критически важен для безопасности
        return yaml.safe_load(f)

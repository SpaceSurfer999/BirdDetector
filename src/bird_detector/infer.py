"""
Модуль инференса BirdDetector.
Загружает обученную модель и делает предсказания на кадрах с камеры.
"""

import torch
from torch import nn

from bird_detector.model import build_model


def load_model(model_path: str, num_classes: int = 2, device: str = "cpu") -> nn.Module:
    """
    Загружает обученную модель из .pth файла.

    Args:
        model_path: Путь к файлу весов (models/best_bird_model.pth).
        num_classes: Количество классов (должно совпадать с обучением!).
        device: 'cpu' или 'cuda'.

    Returns:
        Модель в режиме eval, готовая к предсказаниям.
    """
    # 1. Создаём пустую модель той же архитектуры
    model = build_model(num_classes=2)

    # 2. Загружаем веса из файла (не забудь про weights_only=True и map_location!)
    #    map_location=device важно: если веса сохранены с GPU, а грузим на CPU — без этого упадёт
    state_dict = torch.load(model_path, map_location=device, weights_only=True)

    # 3. Заливаем веса в модель
    model.load_state_dict(state_dict)

    # 4. Переводим на устройство и в режим оценки
    model = model.to(device)
    model.eval()

    return model


if __name__ == "__main__":
    # Тест: загружаем модель и проверяем, что она готова
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model("models/best_bird_model.pth", device=device)

    print(f"Model loaded on {device}")
    print(f"Model in eval mode: {not model.training}")  # Должно быть True

    # Пробный прогон случайного тензора
    dummy = torch.randn(1, 3, 224, 224, device=device)
    with torch.no_grad():  # на инференсе градиенты не нужны — экономим память
        output = model(dummy)
    print(f"Output shape: {output.shape}")  # Ожидается [1, 2]

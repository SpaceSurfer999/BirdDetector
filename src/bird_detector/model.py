"""
Модуль сборки BirdDetector
Использует трансверное обучение ResNet18.
"""

import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18


def build_model(num_classes: int = 2) -> nn.Module:
    """
    Собирает модель для классификации есть ли птица на кормушке.
    Args:
        num_classes: Количество классов
    Returns:
        Готовая модель
    """

    # Загружаем предобученный ResNet18 с весами
    weights = ResNet18_Weights.DEFAULT
    model = resnet18(weights=weights)

    # Замораживаем все параметры кроме головы
    for param in model.parameters():
        param.requires_grad = False

    # Меняем голову модели
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    return model


if __name__ == "__main__":
    # Создаем модель
    m = build_model()

    # Проверяем кол-во парметров для обучения
    trainable_params = sum(p.numel() for p in m.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in m.parameters())

    # Выводим количество. Total ~11 млн, Trainable ~ 1 тыс
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Tест. Должно быть
    dummy_input = torch.rand(1, 3, 224, 224)
    output = m(dummy_input)
    print(f"Output shape: {output.shape}")

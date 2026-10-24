"""
Модуль обучения. Связывает data.py и model.py в пайплайн.
"""

import logging
from pathlib import Path

import torch
from torch import nn

# Импорт модулей проекта
from bird_detector.data import get_dataloaders
from bird_detector.model import build_model

logger = logging.getLogger(__name__)


def train_one_epoch(model, loader, criterion, optimizer, device):
    """
    Одна эпоха обучения.
    Args:
        model: Текущая модель.
        loader: Dataloader для выборки
        criterion: Функция потерь
        optimizer: Оптимизатор
        device: Утройство('cuda', 'cpu')

        Returns:
            tuple: (средний loss за эпоху, средний accuracy за эпоху)
    """
    # Включаем режим обучения
    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    # Цикл обучения и сбора статистики для расчета Accuracy и Loss
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        logits = model(inputs)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
        _, predicted = logits.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


# Декоратором отключаем граф вычислений
@torch.no_grad()
def validate(model, loader, criterion, device):
    """
    Оценка модели на валидационной выборке.

    Args:
        model: Модель для оценки.
        loader: DataLoader для валидации.
        criterion: Функция потерь.
        device: Устройство.

    Returns:
        tuple: (средний val loss, val accuracy)
    """

    # Включаем режим валидации
    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    # Цикл валидации и сбора статистики для расчета Accuracy и Loss
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        logits = model(inputs)
        loss = criterion(logits, targets)
        running_loss += loss.item() * inputs.size(0)
        _, predicted = logits.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    val_loss = running_loss / len(loader.dataset)
    val_acc = correct / total
    return val_loss, val_acc


def run_training(
    epochs: int = 30,
    lr: float = 1e-3,
    batch_size: int = 8,
    save_path: str = "models/best_bird_model.pth",
):
    """
    Главная функция запуска процесса.
    """
    # Настройка устройства
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using {device}")

    # Загрузка данных
    train_loader, val_loader = get_dataloaders(batch_size=batch_size)

    # Инициализация модели
    model = build_model(num_classes=2).to(device)

    # Функция потерь и оптимизатор
    # Для классификации используем CrossEntropyLoss
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
    )

    # Переменные для Early Stopping
    best_val_loss = float("inf")
    best_model_state = None

    logger.info("Starting trainig...")

    for epoch in range(1, epochs + 1):
        # Обучение
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )

        # Валидация
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        logger.info(
            f"Epoch: [{epoch}/{epochs}] |"
            f"Train loss: {train_loss:.4f} Acc: {train_acc:.4f} |"
            f"Val loss: {val_loss:.4f} Acc: {val_acc:.4f}"
        )

        # Проверка на улучшение метрики валидации
        if val_loss < best_val_loss:
            best_val_loss = val_loss

            # Сохраняем состояние модели
            best_model_state = {
                k: v.cpu().clone() for k, v in model.state_dict().items()
            }
            logger.info(f"   >> New best model saved (Val Loss: {best_val_loss:.4f})")

    # Сохранение лучшей модели
    if best_model_state is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(best_model_state, save_path)
        logger.info(f"Training finished. Best model saved to {save_path}")
    else:
        logger.warning("No improvement found during training.")


if __name__ == "__main__":
    # Настройка логирования для запуска скрипта напрямую
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    # Запуск обучения
    # Для начала используем дефолтные значения из сигнатуры функции
    run_training(epochs=20, lr=1e-3, batch_size=8)

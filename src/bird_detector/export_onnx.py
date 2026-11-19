"""
Скрипт экспорта модели BirdDetector в формат ONNX.
Запуск: uv run python -m bird_detector.export_onnx
"""

from pathlib import Path

import torch

# Импортируем нашу функцию загрузки, чтобы не дублировать логику
from bird_detector.infer import load_model


def export_to_onnx(
    model_path: str = "models/best_bird_model.pth",
    output_path: str = "models/bird_detector.onnx",
    opset_version: int = 17,
):
    """
    Конвертирует .pth weights в .onnx graph.
    """

    # 1. Загружаем модель на CPU.
    # Важно: ONNX граф не должен содержать CUDA-операторов, иначе телефон их не поймет.
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    model = load_model(model_path, device="cpu")

    # 2. Переводим в режим оценки.
    # Это отключает Dropout и фиксирует BatchNorm stats. Критично для инференса.
    model.eval()

    # 3. Создаем "фейковый" вход (dummy input).
    # Трассировщик прогонит модель на этом тензоре, чтобы понять размеры всех слоев.
    # Форма: [Batch=1, Channels=3, Height=224, Width=224]
    dummy_input = torch.randn(1, 3, 224, 224)

    print(f"⚙️ Exporting to ONNX (opset {opset_version})...")

    # 4. Сам экспорт
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        dynamo=False,
    )
    file_size = Path(output_path).stat().st_size / (1024 * 1024)
    print(f"Success! Saved to {output_path} ({file_size:.2f} MB)")


if __name__ == "__main__":
    export_to_onnx()

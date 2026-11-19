"""
Скрипт верификации ONNX-экспорта.
Сравнивает логиты PyTorch-модели и ONNX-модели на одном кадре.
"""

import numpy as np
import onnxruntime as ort
import torch

from bird_detector.data import get_transforms
from bird_detector.infer import imread_unicode, load_model, preprocess_frame


def verify_export(
    pth_path: str = "models/best_bird_model.pth",
    onnx_path: str = "models/bird_detector.onnx",
    test_image: str = "data/train/1_bird/frame_20261017_160923.jpg",
    tolerance: float = 1e-4,
):
    """
    Сверяет выводы PyTorch и ONNX Runtime на одном изображении.
    """
    print(f"Verifying ONNX export: {onnx_path}")

    # 1. Загружаем PyTorch модель
    device = "cpu"
    model = load_model(pth_path, device=device)
    transforms = get_transforms(is_train=False)

    # 2. Читаем и препроцессим кадр
    frame = imread_unicode(test_image)
    tensor = preprocess_frame(frame, transforms).to(device)

    # 3. Получаем логиты из PyTorch
    with torch.no_grad():
        torch_logits = model(tensor).cpu().numpy()

    # 4. Получаем логиты из ONNX Runtime
    session = ort.InferenceSession(onnx_path)
    input_name = session.get_inputs()[0].name
    onnx_logits = session.run(None, {input_name: tensor.cpu().numpy()})[0]

    # 5. Сравниваем
    max_diff = np.abs(torch_logits - onnx_logits).max()
    mean_diff = np.abs(torch_logits - onnx_logits).mean()

    print(f"PyTorch logits: {torch_logits}")
    print(f"ONNX logits:    {onnx_logits}")
    print(f"Max difference: {max_diff:.2e}")
    print(f"Mean difference: {mean_diff:.2e}")

    if max_diff < tolerance:
        print(f"PASS: Models match within tolerance {tolerance}")
        return True
    else:
        print(f"FAIL: Models diverge! Max diff {max_diff:.2e} > {tolerance}")
        return False


if __name__ == "__main__":
    verify_export()

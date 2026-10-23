import torch

from bird_detector.data import get_dataloaders


def test_dataloaders():
    print("--- Testing Data Loading ---")
    try:
        train_loader, _val_loader = get_dataloaders(batch_size=4)

        # Проверка формы батча
        images, labels = next(iter(train_loader))
        assert images.shape == (4, 3, 224, 224), f"Wrong shape: {images.shape}"
        assert labels.dtype == torch.int64, (
            f"Labels should be int64, got {labels.dtype}"
        )

        # Проверка диапазона нормализации (должно быть около [-2, 2])
        print(f"Images min/max: {images.min().item():.2f} / {images.max().item():.2f}")
        print(f"Sample labels: {labels.tolist()}")
        print(f"Classes mapping: {train_loader.dataset.class_to_idx}")

        print("✅ All checks passed!")

    except Exception as e:
        print(f"❌ Error: {e}")
        raise


if __name__ == "__main__":
    test_dataloaders()

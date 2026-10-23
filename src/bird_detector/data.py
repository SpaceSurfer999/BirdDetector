"""
Модуль подготовки данных для BirdDetector.
Использует torchvision.datasets.ImageFolder для автоматической разметки
"""

from pathlib import Path

from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder


def get_transforms(is_train: bool = True) -> transforms.Compose:
    """
    Возвращает пайплайн для обучения и валидации.
    Если is_train=True, то включаются аугментации,
    если False, то только Resize и Normalize
    """

    # задаем стандартные статистики
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    if is_train:
        return transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(
                    brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
                ),
                transforms.ToTensor(),
                transforms.Normalize(imagenet_mean, imagenet_std),
            ]
        )
    else:
        return transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(imagenet_mean, imagenet_std),
            ]
        )


class BirdDataset(Dataset):
    """
    Обертка над ImageFolder для удобства интеграции в наш проект.
    """

    def __init__(self, root_dir: str, split: str = "train"):
        """
        root_dir : путь к корню данных.
        split: "train" или "val"
        """
        self.root_path = Path(root_dir) / split

        if not self.root_path.exists():
            raise FileNotFoundError(f"Папка {self.root_path} не найдена")

        self.dataset = ImageFolder(
            root=str(self.root_path),
            transform=get_transforms(is_train=(split == "train")),
        )

        self.classes = self.dataset.classes
        self.class_to_idx = self.dataset.class_to_idx

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]


def get_dataloaders(
    batch_size: int = 8, num_workers: int = 0
) -> tuple[DataLoader, DataLoader]:
    """
    Создает Dataloader для тренировки и валидации
    batch_size: Размер батча
    num_worker: Количество процессов загрузки
    """
    train_ds = BirdDataset(root_dir="data", split="train")
    val_ds = BirdDataset(root_dir="data", split="val")

    print(f"Loaded train dataset: {len(train_ds)} images. Classes: {train_ds.classes}")
    print(f"Loaded val dataset: {len(val_ds)} images. Classes: {val_ds.classes}")

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader

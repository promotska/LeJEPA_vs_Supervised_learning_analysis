from __future__ import annotations

from typing import Callable

from torchvision import transforms

from src.globals import CIFAR10_MEAN, CIFAR10_STD


class TwoCropsTransform:
    """Return two independently augmented views of the same image."""

    def __init__(self, base_transform: Callable):
        self.base_transform = base_transform

    def __call__(self, image):
        return self.base_transform(image), self.base_transform(image)


def build_supervised_train_transform():
    return transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )


def build_eval_transform():
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )


def build_lejepa_train_transform():
    base = transforms.Compose(
        [
            transforms.RandomResizedCrop(32, scale=(0.6, 1.0), ratio=(0.75, 1.3333)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.05),
            transforms.RandomGrayscale(p=0.1),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )
    return TwoCropsTransform(base)

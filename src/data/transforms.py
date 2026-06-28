from __future__ import annotations

from typing import Callable, Sequence

from torchvision import transforms

from src.globals import CIFAR10_MEAN, CIFAR10_STD


def _to_tuple2(value: Sequence[float] | tuple[float, float], default: tuple[float, float]) -> tuple[float, float]:
    if value is None:
        return default
    return (float(value[0]), float(value[1]))


class TwoCropsTransform:
    """Return two independently augmented views of the same image."""

    def __init__(self, base_transform: Callable):
        self.base_transform = base_transform

    def __call__(self, image):
        return self.base_transform(image), self.base_transform(image)


class MultiCropTransform:
    """
    Return a list of global and local crops.

    For CIFAR, local crops are resized back to image_size so they can be batched.
    This is a CIFAR-adapted analogue of the global/local multi-crop recipe.
    """

    def __init__(
        self,
        global_transform: Callable,
        local_transform: Callable,
        num_global_views: int = 2,
        num_local_views: int = 4,
    ):
        if num_global_views < 1:
            raise ValueError("num_global_views must be >= 1")
        if num_local_views < 0:
            raise ValueError("num_local_views must be >= 0")
        self.global_transform = global_transform
        self.local_transform = local_transform
        self.num_global_views = int(num_global_views)
        self.num_local_views = int(num_local_views)

    def __call__(self, image):
        views = [self.global_transform(image) for _ in range(self.num_global_views)]
        views.extend(self.local_transform(image) for _ in range(self.num_local_views))
        return views


def build_supervised_train_transform(image_size: int = 32):
    if image_size != 32:
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(image_size, scale=(0.6, 1.0), ratio=(0.75, 1.3333)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
            ]
        )

    return transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )


def build_eval_transform(image_size: int = 32):
    steps = []
    if image_size != 32:
        steps.extend([transforms.Resize(image_size), transforms.CenterCrop(image_size)])
    steps.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )
    return transforms.Compose(steps)


def _build_crop_transform(
    image_size: int,
    scale: tuple[float, float],
    color_jitter_strength: float = 0.25,
    grayscale_p: float = 0.1,
):
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size, scale=scale, ratio=(0.75, 1.3333)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(
                brightness=color_jitter_strength,
                contrast=color_jitter_strength,
                saturation=color_jitter_strength,
                hue=min(0.1, color_jitter_strength / 5),
            ),
            transforms.RandomGrayscale(p=grayscale_p),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )


def build_lejepa_train_transform(
    image_size: int = 32,
    num_global_views: int = 2,
    num_local_views: int = 0,
    global_scale: Sequence[float] = (0.6, 1.0),
    local_scale: Sequence[float] = (0.2, 0.6),
    color_jitter_strength: float = 0.25,
    grayscale_p: float = 0.1,
):
    """
    Build self-supervised views.

    Backward-compatible behavior:
      num_global_views=2 and num_local_views=0 -> TwoCropsTransform.

    LeJEPA-SIGReg behavior:
      num_global_views=2 and num_local_views>0 -> MultiCropTransform.
    """
    global_scale_t = _to_tuple2(global_scale, (0.6, 1.0))
    local_scale_t = _to_tuple2(local_scale, (0.2, 0.6))

    global_transform = _build_crop_transform(
        image_size=image_size,
        scale=global_scale_t,
        color_jitter_strength=color_jitter_strength,
        grayscale_p=grayscale_p,
    )

    if int(num_local_views) <= 0 and int(num_global_views) == 2:
        return TwoCropsTransform(global_transform)

    local_transform = _build_crop_transform(
        image_size=image_size,
        scale=local_scale_t,
        color_jitter_strength=color_jitter_strength,
        grayscale_p=grayscale_p,
    )

    return MultiCropTransform(
        global_transform=global_transform,
        local_transform=local_transform,
        num_global_views=int(num_global_views),
        num_local_views=int(num_local_views),
    )

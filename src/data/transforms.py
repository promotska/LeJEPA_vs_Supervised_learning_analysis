from __future__ import annotations

from typing import Callable, Sequence

from torchvision import transforms

from src.globals import CIFAR10_MEAN, CIFAR10_STD, CIFAR100_MEAN, CIFAR100_STD, IMAGENET_MEAN, IMAGENET_STD


FloatTriple = tuple[float, float, float]


def _to_tuple2(value: Sequence[float] | tuple[float, float] | None, default: tuple[float, float]) -> tuple[float, float]:
    if value is None:
        return default
    return (float(value[0]), float(value[1]))


def _to_tuple3(value: Sequence[float] | tuple[float, float, float] | None, default: FloatTriple) -> FloatTriple:
    if value is None:
        return default
    return (float(value[0]), float(value[1]), float(value[2]))


def default_normalization(dataset: str) -> tuple[FloatTriple, FloatTriple]:
    name = dataset.lower()
    if name in {"cifar100", "cifar-100"}:
        return CIFAR100_MEAN, CIFAR100_STD
    if name in {"imagefolder", "imagenet100", "imagenet-100", "imagenet1k", "imagenet-1k", "imagenet"}:
        return IMAGENET_MEAN, IMAGENET_STD
    return CIFAR10_MEAN, CIFAR10_STD


class TwoCropsTransform:
    """Return two independently augmented views of the same image."""

    def __init__(self, base_transform: Callable):
        self.base_transform = base_transform

    def __call__(self, image):
        return self.base_transform(image), self.base_transform(image)


class MultiCropTransform:
    """Return official-style global/local crops as a list of tensors.

    The exact crop ranges are configurable. For CIFAR, all crops are resized back
    to 32x32; for ImageNet-style datasets, they are usually resized to 224x224.
    """

    def __init__(
        self,
        global_transform: Callable,
        local_transform: Callable,
        num_global_views: int = 2,
        num_local_views: int = 6,
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


def _normalize(mean: FloatTriple, std: FloatTriple):
    return transforms.Normalize(mean, std)


def build_supervised_train_transform(
    image_size: int = 32,
    mean: FloatTriple = CIFAR10_MEAN,
    std: FloatTriple = CIFAR10_STD,
):
    if image_size != 32:
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(image_size, scale=(0.6, 1.0), ratio=(0.75, 1.3333)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                _normalize(mean, std),
            ]
        )

    return transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            _normalize(mean, std),
        ]
    )


def build_eval_transform(
    image_size: int = 32,
    mean: FloatTriple = CIFAR10_MEAN,
    std: FloatTriple = CIFAR10_STD,
    resize_size: int | None = None,
):
    steps = []
    if image_size != 32:
        resize_size = int(resize_size or round(image_size * 256 / 224))
        steps.extend([transforms.Resize(resize_size), transforms.CenterCrop(image_size)])
    steps.extend([transforms.ToTensor(), _normalize(mean, std)])
    return transforms.Compose(steps)


def _build_crop_transform(
    image_size: int,
    scale: tuple[float, float],
    mean: FloatTriple,
    std: FloatTriple,
    color_jitter_strength: float = 0.4,
    grayscale_p: float = 0.2,
    blur_p: float = 0.0,
):
    ops = [
        transforms.RandomResizedCrop(image_size, scale=scale, ratio=(0.75, 1.3333)),
        transforms.RandomHorizontalFlip(),
    ]

    if color_jitter_strength > 0:
        ops.append(
            transforms.RandomApply(
                [
                    transforms.ColorJitter(
                        brightness=color_jitter_strength,
                        contrast=color_jitter_strength,
                        saturation=color_jitter_strength,
                        hue=min(0.1, color_jitter_strength / 5),
                    )
                ],
                p=0.8,
            )
        )

    if grayscale_p > 0:
        ops.append(transforms.RandomGrayscale(p=grayscale_p))

    if blur_p > 0:
        kernel_size = max(3, int(0.1 * image_size) // 2 * 2 + 1)
        ops.append(transforms.RandomApply([transforms.GaussianBlur(kernel_size=kernel_size, sigma=(0.1, 2.0))], p=blur_p))

    ops.extend([transforms.ToTensor(), _normalize(mean, std)])
    return transforms.Compose(ops)


def build_lejepa_train_transform(
    image_size: int = 32,
    num_global_views: int = 2,
    num_local_views: int = 6,
    global_scale: Sequence[float] = (0.4, 1.0),
    local_scale: Sequence[float] = (0.05, 0.4),
    mean: FloatTriple = CIFAR10_MEAN,
    std: FloatTriple = CIFAR10_STD,
    color_jitter_strength: float = 0.4,
    grayscale_p: float = 0.2,
    global_blur_p: float = 0.1,
    local_blur_p: float = 0.5,
):
    """Build official-style LeJEPA multi-crop views.

    Default is 2 global + 6 local views. The crop scales are intentionally
    configurable so the same code can be used for CIFAR-sized and ImageNet-sized
    experiments. No stop-gradient or teacher branch is introduced by the transform.
    """
    global_scale_t = _to_tuple2(global_scale, (0.4, 1.0))
    local_scale_t = _to_tuple2(local_scale, (0.05, 0.4))

    global_transform = _build_crop_transform(
        image_size=image_size,
        scale=global_scale_t,
        mean=mean,
        std=std,
        color_jitter_strength=color_jitter_strength,
        grayscale_p=grayscale_p,
        blur_p=global_blur_p,
    )

    # Backward-compatible behavior for old Stage 1 debug configs.
    if int(num_local_views) <= 0 and int(num_global_views) == 2:
        return TwoCropsTransform(global_transform)

    local_transform = _build_crop_transform(
        image_size=image_size,
        scale=local_scale_t,
        mean=mean,
        std=std,
        color_jitter_strength=color_jitter_strength,
        grayscale_p=grayscale_p,
        blur_p=local_blur_p,
    )

    return MultiCropTransform(
        global_transform=global_transform,
        local_transform=local_transform,
        num_global_views=int(num_global_views),
        num_local_views=int(num_local_views),
    )

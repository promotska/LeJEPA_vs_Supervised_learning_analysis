from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import VOCSegmentation
from torchvision.transforms import InterpolationMode

from src.data.transforms import default_normalization
from src.utils import seed_worker


IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
MASK_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


@dataclass(frozen=True)
class MaskSample:
    image_path: Path
    mask_path: Path
    label: int
    sample_id: str


class PairedImageMaskTransform:
    """Apply identical deterministic geometry to image and mask.

    The augmented image receives only photometric changes, so it remains pixel-aligned
    with the original image and mask. This makes PCA stability computation simple:
    Corr(PCA(image), PCA(photometric_aug(image))).
    """

    def __init__(
        self,
        image_size: int = 224,
        resize_size: int | None = 256,
        mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: tuple[float, float, float] = (0.229, 0.224, 0.225),
        aug_strength: float = 0.20,
        grayscale_p: float = 0.0,
    ):
        self.image_size = int(image_size)
        self.resize_size = int(resize_size or round(image_size * 256 / 224))
        self.mean = tuple(float(v) for v in mean)
        self.std = tuple(float(v) for v in std)

        self.image_geometry = transforms.Compose(
            [
                transforms.Resize(self.resize_size, interpolation=InterpolationMode.BICUBIC),
                transforms.CenterCrop(self.image_size),
            ]
        )
        self.mask_geometry = transforms.Compose(
            [
                transforms.Resize(self.resize_size, interpolation=InterpolationMode.NEAREST),
                transforms.CenterCrop(self.image_size),
            ]
        )
        if aug_strength > 0:
            self.photometric_aug = transforms.Compose(
                [
                    transforms.ColorJitter(
                        brightness=aug_strength,
                        contrast=aug_strength,
                        saturation=aug_strength,
                        hue=min(0.05, aug_strength / 5.0),
                    ),
                    transforms.RandomGrayscale(p=float(grayscale_p)),
                ]
            )
        else:
            self.photometric_aug = None

        self.to_tensor_norm = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std),
            ]
        )
        self.mask_to_tensor = transforms.PILToTensor()

    def __call__(self, image: Image.Image, mask: Image.Image) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if image.mode != "RGB":
            image = image.convert("RGB")
        if mask.mode not in {"L", "P", "I"}:
            mask = mask.convert("L")

        image_geo = self.image_geometry(image)
        mask_geo = self.mask_geometry(mask)

        if self.photometric_aug is not None:
            image_aug = self.photometric_aug(image_geo)
        else:
            image_aug = image_geo

        image_t = self.to_tensor_norm(image_geo)
        image_aug_t = self.to_tensor_norm(image_aug)
        mask_t = self.mask_to_tensor(mask_geo).squeeze(0).long()
        return image_t, image_aug_t, mask_t


class VOCBinaryMaskDataset(Dataset):
    """VOC semantic segmentation as binary foreground masks.

    Output tuple:
      image, image_aug, label, gt_binary, valid_mask, sample_id

    VOC mask convention:
      0 = background
      1..20 = semantic object classes
      255 = void / ignored pixels
    """

    def __init__(
        self,
        root: str | Path,
        year: str = "2012",
        split: str = "val",
        image_size: int = 224,
        resize_size: int | None = 256,
        mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: tuple[float, float, float] = (0.229, 0.224, 0.225),
        aug_strength: float = 0.20,
        grayscale_p: float = 0.0,
    ):
        self.dataset = VOCSegmentation(
            root=str(root),
            year=str(year),
            image_set=str(split),
            download=False,
        )
        self.transform = PairedImageMaskTransform(
            image_size=image_size,
            resize_size=resize_size,
            mean=mean,
            std=std,
            aug_strength=aug_strength,
            grayscale_p=grayscale_p,
        )

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        image, mask = self.dataset[index]
        image_t, image_aug_t, mask_t = self.transform(image, mask)
        valid_mask = mask_t != 255
        gt_binary = ((mask_t > 0) & valid_mask).float()
        return image_t, image_aug_t, -1, gt_binary, valid_mask.float(), str(index)


class FolderBinaryMaskDataset(Dataset):
    """Generic image/mask folder dataset for ImageNet-S-like masks.

    Expected pairing is by filename stem. This supports common layouts such as:
      image_root/class_x/n01440764_18.JPEG
      mask_root/class_x/n01440764_18.png

    or flat directories:
      image_root/n01440764_18.JPEG
      mask_root/n01440764_18.png

    Masks are converted to binary foreground by default:
      foreground: mask value > 0 and not ignored
      background: mask value == 0
      ignored: mask value == ignore_index, default 255
    """

    def __init__(
        self,
        image_root: str | Path,
        mask_root: str | Path,
        split_file: str | Path | None = None,
        image_size: int = 224,
        resize_size: int | None = 256,
        mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: tuple[float, float, float] = (0.229, 0.224, 0.225),
        ignore_index: int = 255,
        aug_strength: float = 0.20,
        grayscale_p: float = 0.0,
        use_subfolder_labels: bool = True,
    ):
        self.image_root = Path(image_root)
        self.mask_root = Path(mask_root)
        self.ignore_index = int(ignore_index)
        if not self.image_root.exists():
            raise FileNotFoundError(f"image_root does not exist: {self.image_root}")
        if not self.mask_root.exists():
            raise FileNotFoundError(f"mask_root does not exist: {self.mask_root}")

        self.transform = PairedImageMaskTransform(
            image_size=image_size,
            resize_size=resize_size,
            mean=mean,
            std=std,
            aug_strength=aug_strength,
            grayscale_p=grayscale_p,
        )

        self.class_to_idx: dict[str, int] = {}
        if use_subfolder_labels:
            classes = sorted(p.name for p in self.image_root.iterdir() if p.is_dir())
            self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(classes)}

        mask_by_key = self._index_masks(self.mask_root)
        image_paths = self._list_images(split_file)

        samples: list[MaskSample] = []
        missing_masks: list[str] = []
        for image_path in image_paths:
            rel = image_path.relative_to(self.image_root)
            key_candidates = [
                str(rel.with_suffix("")),
                image_path.stem,
            ]
            mask_path = None
            for key in key_candidates:
                if key in mask_by_key:
                    mask_path = mask_by_key[key]
                    break
            if mask_path is None:
                missing_masks.append(str(rel))
                continue

            label = -1
            if use_subfolder_labels and len(rel.parts) > 1:
                label = self.class_to_idx.get(rel.parts[0], -1)

            samples.append(
                MaskSample(
                    image_path=image_path,
                    mask_path=mask_path,
                    label=label,
                    sample_id=str(rel.with_suffix("")),
                )
            )

        if not samples:
            preview = "\n".join(missing_masks[:10])
            raise RuntimeError(
                f"No image/mask pairs found. image_root={self.image_root} mask_root={self.mask_root}\n"
                f"First missing examples:\n{preview}"
            )

        if missing_masks:
            print(f"WARNING: {len(missing_masks)} images had no matching mask and were skipped.")

        self.samples = samples

    @staticmethod
    def _index_masks(mask_root: Path) -> dict[str, Path]:
        out: dict[str, Path] = {}
        for path in mask_root.rglob("*"):
            if path.is_file() and path.suffix.lower() in MASK_EXTENSIONS:
                rel = path.relative_to(mask_root)
                out[str(rel.with_suffix(""))] = path
                out[path.stem] = path
        return out

    def _list_images(self, split_file: str | Path | None) -> list[Path]:
        if split_file is None:
            return sorted(
                p for p in self.image_root.rglob("*")
                if p.is_file() and p.suffix.lower() in IMG_EXTENSIONS
            )

        split_path = Path(split_file)
        if not split_path.exists():
            raise FileNotFoundError(f"split_file does not exist: {split_path}")

        paths: list[Path] = []
        for line in split_path.read_text(encoding="utf-8").splitlines():
            item = line.strip().split()[0] if line.strip() else ""
            if not item:
                continue
            candidate = self.image_root / item
            if candidate.suffix.lower() not in IMG_EXTENSIONS:
                matches = [p for p in self.image_root.rglob(f"{item}.*") if p.suffix.lower() in IMG_EXTENSIONS]
                if matches:
                    candidate = matches[0]
            if candidate.exists():
                paths.append(candidate)
            else:
                raise FileNotFoundError(f"Image listed in split file not found: {candidate}")
        return paths

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        image = Image.open(sample.image_path).convert("RGB")
        mask = Image.open(sample.mask_path)
        image_t, image_aug_t, mask_t = self.transform(image, mask)

        valid_mask = mask_t != self.ignore_index
        gt_binary = ((mask_t > 0) & valid_mask).float()
        return image_t, image_aug_t, int(sample.label), gt_binary, valid_mask.float(), sample.sample_id


def build_mask_loader(cfg: dict[str, Any]) -> DataLoader:
    gcfg = cfg.get("grounded_evaluation", {})
    if not gcfg:
        raise KeyError("Config must contain grounded_evaluation for G-LSAS.")

    dataset_name = str(gcfg.get("dataset", "voc2012")).lower()
    data_cfg = cfg.get("data", {})
    model_cfg = cfg.get("model", {})
    image_size = int(gcfg.get("image_size", model_cfg.get("image_size", data_cfg.get("image_size", 224))))
    resize_size = gcfg.get("resize_size", data_cfg.get("resize_size", 256))
    resize_size = None if resize_size in {None, "none", "None", "null"} else int(resize_size)

    norm_name = str(gcfg.get("normalization", data_cfg.get("dataset", "imagenet100")))
    mean, std = default_normalization(norm_name)
    mean = tuple(float(v) for v in gcfg.get("mean", data_cfg.get("mean", mean)))
    std = tuple(float(v) for v in gcfg.get("std", data_cfg.get("std", std)))

    common = dict(
        image_size=image_size,
        resize_size=resize_size,
        mean=mean,
        std=std,
        aug_strength=float(gcfg.get("stability_aug_strength", 0.20)),
        grayscale_p=float(gcfg.get("stability_grayscale_p", 0.0)),
    )

    if dataset_name in {"voc", "voc2012", "pascal_voc", "pascal-voc"}:
        dataset: Dataset = VOCBinaryMaskDataset(
            root=gcfg.get("root", "datasets"),
            year=str(gcfg.get("year", "2012")),
            split=str(gcfg.get("split", "val")),
            **common,
        )
    elif dataset_name in {"folder", "folder_masks", "imagenet_s", "imagenet-s", "imagenet_s_folder"}:
        dataset = FolderBinaryMaskDataset(
            image_root=gcfg.get("image_root", gcfg.get("images_root", "datasets/imagenet_s/images/val")),
            mask_root=gcfg.get("mask_root", gcfg.get("masks_root", "datasets/imagenet_s/masks/val")),
            split_file=gcfg.get("split_file", None),
            ignore_index=int(gcfg.get("ignore_index", 255)),
            use_subfolder_labels=bool(gcfg.get("use_subfolder_labels", True)),
            **common,
        )
    else:
        raise ValueError(f"Unsupported grounded_evaluation.dataset={dataset_name!r}")

    num_workers = int(gcfg.get("num_workers", data_cfg.get("num_workers", 4)))
    return DataLoader(
        dataset,
        batch_size=int(gcfg.get("batch_size", cfg.get("evaluation", {}).get("batch_size", 16))),
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=seed_worker,
        persistent_workers=num_workers > 0,
    )

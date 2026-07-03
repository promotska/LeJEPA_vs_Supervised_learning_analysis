from __future__ import annotations

import argparse
from pathlib import Path

IMG_EXT = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
MASK_EXT = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}


def count_files(root: Path, exts: set[str]) -> int:
    if not root.exists():
        return 0
    return sum(1 for p in root.rglob('*') if p.is_file() and p.suffix.lower() in exts)


def list_classes(root: Path) -> list[str]:
    if not root.exists():
        return []
    return sorted(p.name for p in root.iterdir() if p.is_dir())


def main() -> None:
    parser = argparse.ArgumentParser(description='Check ImageFolder + mask layout for ImageNet-100/ImageNet-S stages.')
    parser.add_argument('--classification-root', required=True, help='Root with train/ and val/ subfolders')
    parser.add_argument('--mask-root', default=None, help='Optional mask root, usually <dataset>/masks/val')
    args = parser.parse_args()

    root = Path(args.classification_root)
    train = root / 'train'
    val = root / 'val'

    print(f'classification_root: {root}')
    print(f'train exists: {train.exists()}')
    print(f'val exists:   {val.exists()}')

    train_classes = list_classes(train)
    val_classes = list_classes(val)
    print(f'train classes: {len(train_classes)}')
    print(f'val classes:   {len(val_classes)}')
    print(f'train images:  {count_files(train, IMG_EXT)}')
    print(f'val images:    {count_files(val, IMG_EXT)}')

    missing_val = sorted(set(train_classes) - set(val_classes))[:20]
    extra_val = sorted(set(val_classes) - set(train_classes))[:20]
    if missing_val:
        print(f'WARNING: classes in train but not val, first 20: {missing_val}')
    if extra_val:
        print(f'WARNING: classes in val but not train, first 20: {extra_val}')

    if args.mask_root:
        mask_root = Path(args.mask_root)
        print(f'mask_root: {mask_root}')
        print(f'mask exists: {mask_root.exists()}')
        print(f'masks: {count_files(mask_root, MASK_EXT)}')
        mask_classes = list_classes(mask_root)
        print(f'mask classes: {len(mask_classes)}')
        missing_masks = sorted(set(val_classes) - set(mask_classes))[:20]
        if missing_masks:
            print(f'WARNING: val classes missing from mask root, first 20: {missing_masks}')

    if not train.exists() or not val.exists() or not train_classes or not val_classes:
        raise SystemExit('Dataset layout is not ready for ImageFolder training.')


if __name__ == '__main__':
    main()

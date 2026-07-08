from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="OK-AI/lejepa-vitb16-pretrain-in1k")
    parser.add_argument("--revision", default=None)
    parser.add_argument("--trust-remote-code", action="store_true")
    args = parser.parse_args()

    try:
        from transformers import AutoImageProcessor, AutoModel
    except ImportError as exc:
        raise ImportError("Install with: pip install transformers safetensors") from exc

    kwargs = {"trust_remote_code": args.trust_remote_code}
    if args.revision:
        kwargs["revision"] = args.revision

    print(f"Downloading/caching model: {args.model_id} revision={args.revision or 'main'}")
    AutoModel.from_pretrained(args.model_id, **kwargs)
    try:
        AutoImageProcessor.from_pretrained(args.model_id, **kwargs)
    except Exception as exc:
        print(f"Processor download skipped/failed: {exc!r}")
    print("Done. You can now run with HF_HUB_OFFLINE=1 and hf_local_files_only: true.")


if __name__ == "__main__":
    main()

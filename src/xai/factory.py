from __future__ import annotations


def get_vit_xai_method(cfg: dict) -> str:
    return str(cfg.get("evaluation", {}).get("vit_xai_method", "token_gradient")).lower()


def is_vit_xai_method_supported(method: str) -> bool:
    return method.lower() in {"token_gradient", "attention_rollout"}

"""Qwen3-VL backbone used by the released models."""


def get_vlm_model(config):
    from .QWen3 import _QWen3_VL_Interface

    return _QWen3_VL_Interface(config)

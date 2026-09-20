"""Frozen visual features for future latent alignment."""


def get_latent_model(config):
    if config.framework.latent_extractor.latent_type != "vjepa":
        raise ValueError("This release supports latent_type=vjepa (V-JEPA 2).")
    from .vjepa_extractor import VJEPA_Extractor

    return VJEPA_Extractor(config)

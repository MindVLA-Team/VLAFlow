import torch
from transformers import AutoModel, AutoVideoProcessor


class VJEPA_Extractor(torch.nn.Module):
    def __init__(self, global_config):
        super().__init__()
        action_config = global_config.framework.action_model
        extractor_config = global_config.framework.latent_extractor

        # VLM hidden size (used only for KV sharing — action expert can differ)
        self.hidden_size = global_config.framework.qwenvl.vl_hidden_dim

        # Determine action expert internal hidden dim.
        # Priority: explicit action_hidden_dim > hidden_dim_factor * vl_hidden_dim > vl_hidden_dim (default 1.0)
        _factor = float(getattr(action_config, "hidden_dim_factor", 1.0))
        _explicit = getattr(action_config, "action_hidden_dim", None)
        if _explicit is not None:
            action_hidden_dim = int(_explicit)
        else:
            action_hidden_dim = int(self.hidden_size * _factor)

        encoder_path = extractor_config.encoder_path
        self.processor = AutoVideoProcessor.from_pretrained(encoder_path)
        self.encoder = AutoModel.from_pretrained(encoder_path)

        self.output_dim = self.encoder.config.hidden_size
        # Do not need the linear projector, because it may tend to project features into constant to lower the loss
        # the linear projector will be set in action modules

    def get_extractor_dim(self):
        return self.output_dim

    def get_future_latent(self, future_images):
        # future_images [B, [PIL]]
        device = next(self.encoder.parameters()).device
        pixel_values = self.processor(future_images, return_tensors="pt").to(device)["pixel_values_videos"]
        with torch.no_grad():
            vjepa_latents = self.encoder.get_vision_features(pixel_values)  # [B, T, H]

        return vjepa_latents

    def forward(self, future_images):
        return self.get_future_latent(future_images=future_images)

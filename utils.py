import torch
import numpy as np

from PIL import Image

@torch.no_grad()
def latent2image(latents, vae, return_type="np"):
    latents = 1 / 0.18215 * latents.detach()
    image = vae.decode(latents)['sample']
    if return_type == "np":
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()
        image = (image * 255).astype(np.uint8)
    return image
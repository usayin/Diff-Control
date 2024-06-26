import torch
import torch.nn as nn

from StableDiffusionControlnetPipeline import StableDiffusionControlnetPipeline
from Unet import UNet
from typing import List, Optional

class StegoPipeline(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.diffusion_pipeline = StableDiffusionControlnetPipeline(train_mode=True)
        self.decoder = UNet(in_channels=3, out_channels=3)

    
    def forward(self,
                x: torch.Tensor,
                prompts: List[str],
                negative_prompts: Optional[List[str]],
                seed: Optional[int]):
        
        stego_images = self.diffusion_pipeline(
            secret_inputs=x,
            prompts=prompts,
            negative_prompts=negative_prompts,
            seed=seed,
        )

        return self.decoder(stego_images)


if __name__ == "__main__":
    secret_input = torch.randn((1, 3, 512, 512)).to("cuda")
    prompts = ["a cat"]
    negative_prompts = [""]
    seed = 1

    model = StegoPipeline()    
    model.train()

    model.decoder.to("cuda")

    decoded_output = model(
        secret_input,
        prompts=prompts,
        negative_prompts=negative_prompts,
        seed=1
    )

    print(secret_input.shape)
    print(decoded_output.shape)

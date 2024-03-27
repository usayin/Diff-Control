import torch

import numpy as np
import torch.nn as nn

from diffusers import StableDiffusionPipeline, DDIMScheduler, ControlNetModel
from typing import Union, List, Optional
from tqdm import tqdm
from PIL import Image
from icecream import ic


def zero_module(model: nn.Module):
    for parameter in model.parameters():
        nn.init.zeros_(parameter)

    return model


class ControlInputConv(nn.Module):
    """This module is to replace the Conditional Embeddings of ControlNet to handle the message input
    """
    def __init__(self):
        super().__init__()
        # out_channel 320 
        self.conv_in = nn.Conv2d(3, 320, 1)

    def forward(self, x):
        return self.conv_in(x)


class StableDiffusionControlnetPipeline(nn.Module):
    def __init__(self, 
                 pipeline_path: str="runwayml/stable-diffusion-v1-5", 
                 controlnet_path: str="lllyasviel/sd-controlnet-canny",
                 num_inference_steps: int=50,
                 guidance_scale: float=7.5,
                 width: int=512,
                 height: int=512,
                 train_mode: bool=True,) -> None:

        super(StableDiffusionControlnetPipeline, self).__init__()
        self.is_train = train_mode
        self.model_path = pipeline_path
        self.controlnet_path = controlnet_path
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale
        self.sample_width = width
        self.sample_height = height
        self._my_token= "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"
        self.load_models()


    def DDIMLoop(self, 
                 latents: torch.Tensor,
                 encoder_hidden_states: torch.Tensor,
                 secret_inputs: torch.Tensor,
                 timesteps: List[int]) -> torch.Tensor:

        for index, t in tqdm(enumerate(timesteps), total=len(timesteps)):
            input_latents = torch.concat([latents] * 2)

            if index == len(timesteps) - 1:
                # add controlnet outputs to unet for noise predicting
                secret_inputs = torch.cat([secret_inputs] * 2)

                down_block_res_samples, mid_block_res_sample = self.controlnet(
                    input_latents,
                    t,
                    encoder_hidden_states=encoder_hidden_states,
                    controlnet_cond=secret_inputs, # shape=(batch_size, channels, height, width)
                    conditioning_scale=1.0,
                    guess_mode=False,
                    return_dict=False,
                )

                noise_pred = self.unet(
                    input_latents,
                    t,
                    encoder_hidden_states=encoder_hidden_states,
                    down_block_additional_residuals=down_block_res_samples,
                    mid_block_additional_residual=mid_block_res_sample,
                ).sample
            else:
                with torch.no_grad():
                    noise_pred = self.unet(input_latents, t, encoder_hidden_states=encoder_hidden_states).sample

            # preform guidance
            noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_cond - noise_pred_uncond)

            latents = self.scheduler.step(noise_pred, t, latents).prev_sample

        return latents


    def turn_on_train_mode(self):
        self.vae.requires_grad_(False) 
        self.unet.requires_grad_(False)
        self.text_encoder.requires_grad_(False)
        self.controlnet.train()
        self.controlnet.enable_gradient_checkpointing() # to save gpu memory


    def encode_prompt(self, prompts: Union[List[str]]):
        text_input = self.tokenizer(prompts,
                                    padding="max_length",
                                    max_length=self.tokenizer.model_max_length,
                                    truncation=True,
                                    return_tensors="pt")

        text_embeddings = self.text_encoder(text_input.input_ids.to(device_type()))[0]

        return text_embeddings


    def prepare_prompts(self, prompts: List[str], negative_prompts: List[str]) -> torch.Tensor:
        text_prompts_embeddings = self.encode_prompt(prompts)
        negative_prompts_embeddings = self.encode_prompt(negative_prompts)

        return torch.concat([negative_prompts_embeddings, text_prompts_embeddings])


    def prepare_latents(self, seed: int, batch_size: int) -> torch.Tensor:
        return torch.randn(
            (batch_size, self.unet.config.in_channels, self.sample_height // 8, self.sample_width // 8),
            generator=torch.manual_seed(seed)
        ).to(device_type()) 


    def retrieve_steps(self) -> List[int]:
        self.scheduler.set_timesteps(self.num_inference_steps)

        return self.scheduler.timesteps


    def decode_latents(self, latents: torch.Tensor, return_type="pt") -> Union[torch.Tensor, np.ndarray]:
        latents = 1 / self.vae.config.scaling_factor * latents
        images = self.vae.decode(latents, return_dict=False)[0]
        images = (images / 2 + 0.5).clamp(0, 1) * 255.0
        if return_type == "pt":
            # don't need to permute when training
            images = images.to(torch.uint8)
        return images.float()

         
    def forward(self,
                secret_inputs: torch.FloatTensor,
                prompts: List[str],
                negative_prompts: Optional[List[str]],
                seed: int,) -> torch.Tensor:
        batch_size = len(prompts)

        if negative_prompts is None:
            negative_prompts = [""] * len(prompts)

        # prepare latents
        latents = self.prepare_latents(seed=seed, batch_size=batch_size)

        # prepare embeddings
        # TODO: Lazy transfer, Try hooks to optimize redundent code below
        self.text_encoder.to(device_type())
        embeddings = self.prepare_prompts(prompts=prompts, negative_prompts=negative_prompts)
        self.text_encoder.cpu()

        # prepare timesteps
        timesteps = self.retrieve_steps()

        # do inference
        self.unet.to(device_type())
        self.controlnet.to(device_type())
        sample = self.DDIMLoop(latents, embeddings, secret_inputs, timesteps)
        self.unet.cpu()

        # decode samples 
        self.vae.to(device_type())
        images = self.decode_latents(sample, return_type="pt")
        self.vae.cpu()

        return images


    def move_to_gpu(self) -> None:
        self.unet.to(device_type())
        self.vae.to(device_type())
        self.text_encoder.to(device_type())
        self.controlnet.to(device_type())


    def load_models(self) -> None:
        pipeline = StableDiffusionPipeline.from_pretrained(
            self.model_path,
            use_auth_token=self._my_token
        )

        self.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)

        # split pipieline into basic models
        self.unet = pipeline.unet
        self.vae = pipeline.vae
        self.tokenizer = pipeline.tokenizer
        self.text_encoder = pipeline.text_encoder

        # load controlnet
        if self.is_train:
            # if trian, load a controlnet which is copied from unet
            self.controlnet = ControlNetModel.from_unet(self.unet)
            self.controlnet.controlnet_cond_embedding = zero_module(ControlInputConv())
        else:
            # load pretrained controlnet
            # TODO: load controlnet from pretrained pt file
            self.controlnet = ControlNetModel.from_pretrained(self.controlnet_path)


def device_type() -> torch.device:
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


if __name__ == "__main__":
    secret_inputs = torch.randint(1, size=(2, 3, 64, 64)).float().to(device_type())

    Pipeline = StableDiffusionControlnetPipeline()

    Pipeline.turn_on_train_mode()

    images = Pipeline(
        prompts=["A monkey perched on a tree branch, gazing at the world below.",
                 "a lion."],
        negative_prompts=["low quality, ugly, bad proportions",
                          "low quality, ugly, bad proportions"],
        secret_inputs=secret_inputs,
        seed=7,
    )

    ic(f"images shape is {images.shape}")
    batch_size = images.shape[0]
    images = images.permute(0, 2, 3, 1).type(torch.uint8)
    images = images.detach().cpu().numpy()
    images = np.split(images, batch_size)
    images = [Image.fromarray(np.squeeze(image)) for image in images]

    for index, image in enumerate(images):
        image.save(f"{index + 1}.png")

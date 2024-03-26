import os
import torch
import argparse
import numpy as np

from PIL import Image
from tqdm import tqdm
from icecream import ic
from diffusers import StableDiffusionPipeline, ControlNetModel, DDIMScheduler

from utils import latent2image


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, nargs='+', default=["A cat"])
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--num_inference_step", type=int, default=50)
    parser.add_argument("--uncond_generation", action="store_true")
    parser.add_argument("--result_dir", type=str, default="./results")

    args = parser.parse_args()

    result_dir = os.path.join(args.result_dir, str(args.num_inference_step))
    ic(result_dir)

    if not os.path.exists(result_dir):
        os.makedirs(result_dir, exist_ok=True)

    """
        Load basic pretrained models to do inference

        The following models are required to preform controlnet guided diffusion generation using DDIM scheduler
        1. U-Net
        2. vae
        3. controlnet
        4. CLIP's tokenier and text-encoder
        5. DDIM scheduler
    """
    my_token = "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"
    stable_diffuion_repo_id = "runwayml/stable-diffusion-v1-5"
    controlnet_repo_id = "lllyasviel/sd-controlnet-canny"

    pipeline = StableDiffusionPipeline.from_pretrained(
        stable_diffuion_repo_id,
        use_auth_token=my_token,
        use_safetensors=True
    )

    controlnet = ControlNetModel.from_pretrained(
        controlnet_repo_id,
        use_auth_token=my_token,
        use_safetensors=True
    )

    scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)

    # move all models to GPU
    device_type = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    pipeline.to(device_type)
    controlnet.to(device_type)


    """Basic inference stage's arguments"""
    prompt = args.prompt
    guidance_scale = args.guidance_scale
    num_inference_step = args.num_inference_step
    ic(prompt, guidance_scale, num_inference_step)

    generator = torch.manual_seed(666)
    height, width = 512, 512
    batch_size = len(prompt)

    """Get text embeddings for the passed prompt"""
    text_input = pipeline.tokenizer(prompt,
                                    padding="max_length",
                                    max_length=pipeline.tokenizer.model_max_length,
                                    truncation=True,
                                    return_tensors="pt")

    text_embeddings = pipeline.text_encoder(text_input.input_ids.to(device_type))[0]

    null_input = pipeline.tokenizer([""] * batch_size,
                                    padding="max_length",
                                    max_length=pipeline.tokenizer.model_max_length,
                                    truncation=True,
                                    return_tensors="pt")

    null_embeddings = pipeline.text_encoder(null_input.input_ids.to(device_type))[0]

    embeddings = torch.concat([null_embeddings, text_embeddings]) if args.uncond_generation == False else null_embeddings

    """Generate the initial random noise"""
    latents = torch.randn(
        (batch_size, pipeline.unet.config.in_channels, height // 8, width // 8),
        generator=generator,
    )
    latents = latents.to(device_type)

    """Set scheduler""" 
    scheduler.set_timesteps(num_inference_step)
    
    ic(args.uncond_generation)

    """DDIM Loop"""
    for t in tqdm(scheduler.timesteps):
        latent_model_input = latents if args.uncond_generation == True else torch.concat([latents] * 2)

        with torch.no_grad():
            noise_pred = pipeline.unet(latent_model_input, t, encoder_hidden_states=embeddings).sample
            
        if not args.uncond_generation:
            noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
            # preform guidance
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)

        # compute the previous noisy sample x_t -> x_{t-1}
        latents = scheduler.step(noise_pred, t, latents).prev_sample


    # save image to result dir
    images = latent2image(latents, pipeline.vae)
    ic(images.shape)
    images = np.split(images, batch_size, axis=0)
    images = [np.squeeze(image, axis=0) for image in images]
    for index, image in enumerate(images):
        image = Image.fromarray(image)
        image.save(f"{result_dir}/{prompt[index]}_{num_inference_step}_steps.png")

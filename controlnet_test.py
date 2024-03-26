import cv2
import os
import torch
import argparse

import numpy as np

from PIL import Image
from icecream import ic
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, DDIMScheduler
from diffusers.utils import load_image, make_image_grid

# TODO: enable json file as image input source
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_inference_steps", type=int, default=100)
    parser.add_argument("--prompt", type=str, default="a cat")
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--conditional_image_path", type=str, default="https://hf.co/datasets/huggingface/documentation-images/resolve/main/diffusers/input_image_vermeer.png")
    parser.add_argument("--pre_process_image", action="store_true", default=False)
    parser.add_argument("--save_folder", type=str, default="./results/controlTest/")
    
    args = parser.parse_args()
    ic(args)

    url = "/root/autodl-tmp/bathroom.jpg"

    # download an image
    # image = load_image(
    #     image=url
    # )
    image = Image.open(url)

    if args.pre_process_image:
        image = np.array(image)

        # get canny image
        image = cv2.Canny(image, 100, 200)
        image = image[:, :, None]
        image = np.concatenate([image, image, image], axis=2)
        image = Image.fromarray(image)

    # load control net and stable diffusion v1-5
    controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny", torch_dtype=torch.float16)
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5", controlnet=controlnet, torch_dtype=torch.float16
    )

    # speed up diffusion process with faster scheduler and memory optimization
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

    pipe.enable_model_cpu_offload()

    save_folder = args.save_folder
    num_inference_steps = args.num_inference_steps
    prompt = args.prompt

    save_path = os.path.join(save_folder, str(num_inference_steps), "bathroom", prompt)

    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)

    # generate image
    control_guidance_start = [0.99, 0.98, 0.97, 0.96, 0.95, 0.94, 0.93, 0.92, 0.90,
                            0.85, 0.80, 0.75, 0.70, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0]
    control_guidance_start = reversed(control_guidance_start)
    images = []
    for start in control_guidance_start:
        generator = torch.manual_seed(5)
        image = pipe(
            prompt, num_inference_steps=100, generator=generator, image=image, control_guidance_start=start, control_guidance_end=1.0,
            guess_mode=False, guidance_scale=7.5, height=512, width=512,
        ).images[0]
        images.append(image)

    for start, image in zip(control_guidance_start, images):
        ic("save_single_image")
        image.save(f"{save_path}/control_start_{start}_process_{args.pre_process_image}.png")

    image_grid = make_image_grid(images, 4, 5)
    image_grid.save(f"{save_path}/result_precess_{args.pre_process_image}.png")


if __name__ == "__main__":
    main()
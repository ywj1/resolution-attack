import os
import time
import argparse
from typing import Optional, Union, Tuple

import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import torchvision.transforms as tvt
from torchvision.transforms import GaussianBlur
import torchvision.transforms as transforms

from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import retrieve_timesteps
from diffusers import (
    StableDiffusionPipeline,
    ControlNetModel,
    StableDiffusionControlNetPipeline,
    DDIMScheduler,
    DDIMInverseScheduler,
)
from diffusers.utils import load_image
from transformers import DPTForDepthEstimation, DPTFeatureExtractor
from controlnet_aux import PidiNetDetector

device = torch.device("cuda:0")


# Define a Gaussian blur class
class GaussianBlur2F:
    def __init__(self, kernel_size: int = 3, sigma: float = 21.0):
        self.GB = GaussianBlur(kernel_size, sigma)

    def FHigh(self, noise: torch.Tensor) -> torch.Tensor:
        return noise - self.GB(noise)

    def FLow(self, noise: torch.Tensor) -> torch.Tensor:
        return self.GB(noise)


# Function to load an image and resize it to the target size
def load_image1(imgname: str, target_size: Optional[Union[int, Tuple[int, int]]] = None) -> torch.Tensor:
    pil_img = Image.open(imgname).convert('RGB')
    if target_size is not None:
        if isinstance(target_size, int):
            target_size = (target_size, target_size)
        pil_img = pil_img.resize(target_size, Image.Resampling.LANCZOS)
    return tvt.ToTensor()(pil_img)[None, ...]  # add batch dimension


# Convert an image tensor to latent representation using the VAE
def img_to_latents(x: torch.Tensor, vae) -> torch.Tensor:
    x = 2. * x - 1.
    posterior = vae.encode(x).latent_dist
    latents = posterior.mean * 0.18215
    return latents


# Visualize the latent channels
def visualize_latents(latents: torch.Tensor):
    latents_np = latents.detach().cpu().numpy()
    num_channels = latents_np.shape[1]
    fig, axes = plt.subplots(1, num_channels, figsize=(20, 4))
    for i in range(num_channels):
        axes[i].imshow(latents_np[0, i], cmap='gray')
        axes[i].axis('off')
        axes[i].set_title(f'Channel {i}')
    plt.show()


# DDIM inversion function
@torch.no_grad()
def ddim_inversion(imgname: str, num_steps: int = 50, prompt: str = "", verify: Optional[bool] = False,
                   model_id: str = "", img_resolution: int = 512) -> torch.Tensor:
    dtype = torch.float16
    inverse_scheduler = DDIMInverseScheduler.from_pretrained(model_id, subfolder='scheduler')
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        scheduler=inverse_scheduler,
        safety_checker=None,
        torch_dtype=dtype
    )
    pipe.to(device)
    vae = pipe.vae

    input_img = load_image1(imgname, target_size=(img_resolution, img_resolution)).to(device=device, dtype=dtype)
    latents = img_to_latents(input_img, vae)

    inv_latents, _ = pipe(
        prompt=prompt,
        negative_prompt="",
        guidance_scale=1.0,
        width=input_img.shape[-1],
        height=input_img.shape[-2],
        output_type='latent',
        return_dict=False,
        num_inference_steps=num_steps,
        latents=latents
    )

    if verify:
        pipe.scheduler = DDIMScheduler.from_pretrained(model_id, subfolder='scheduler')
        image = pipe(
            prompt="",
            negative_prompt="",
            guidance_scale=1.0,
            num_inference_steps=num_steps,
            latents=inv_latents
        )
        fig, ax = plt.subplots(1, 2)
        ax[0].imshow(tvt.ToPILImage()(input_img[0]))
        ax[1].imshow(image.images[0])
        plt.show()
    return inv_latents


def main(args):
    # Set image resolution from args
    img_resolution = args.img_resolution

    # Load the annotators model for generating softedge
    processor = PidiNetDetector.from_pretrained(args.annotators_model_path).to(device)

    # Initialize the Gaussian blur class
    view = GaussianBlur2F()

    # Load ControlNet model for softedge
    controlnet = ControlNetModel.from_pretrained(
        args.model_control_softedge_path,
        torch_dtype=torch.float16
    ).to(device)

    # Initialize the Stable Diffusion ControlNet pipeline
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        args.model_id,
        controlnet=controlnet,
        torch_dtype=torch.float16,
        variant="fp16"
    ).to(device)

    negative_prompt = "Sticking tongue out, blurry, low resolution, unrealistic, distorted"

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Load the input image using the provided image path
    input_image = Image.open(args.input_image_path).convert("RGB")
    # Generate control image using the processor and resize it
    control_image = processor(input_image, safe=True)
    control_image = control_image.resize((img_resolution, img_resolution), Image.Resampling.LANCZOS).convert("RGB")
    transform = transforms.Compose([
        transforms.ToTensor(),  # Convert image to tensor
    ])
    control_image_tensor = transform(control_image).unsqueeze(0).to(device)
    control_image_tensor = control_image_tensor.half()

    # For each high-resolution prompt in the list
    for j, prompt1 in enumerate(args.prompt_list):
        # Perform DDIM inversion to get latent representation from the source image
        latents = ddim_inversion(
            args.input_image_path,
            num_steps=200,
            prompt="",
            verify=False,
            model_id=args.model_id,
            img_resolution=img_resolution
        )

        # Encode the high-resolution and low-resolution prompts
        prompt_embeds_fact1, _ = pipe.encode_prompt(
            prompt1,
            device=device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=True
        )
        prompt_embeds_fact2, _ = pipe.encode_prompt(
            args.prompt2,
            device=device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=True
        )

        # Encode the negative prompt
        neg_prompt_embeds_fact, _ = pipe.encode_prompt(
            negative_prompt,
            device=device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=True
        )

        # Combine positive and negative prompt embeddings for classifier-free guidance
        prompt_embeds_fact1 = torch.cat([neg_prompt_embeds_fact, prompt_embeds_fact1])
        prompt_embeds_fact2 = torch.cat([neg_prompt_embeds_fact, prompt_embeds_fact2])

        # Retrieve timesteps for the scheduler
        timesteps, _ = retrieve_timesteps(pipe.scheduler, 300, device, None)

        with torch.no_grad():
            for t in timesteps:
                latent_model_input = torch.cat([latents] * 2)
                latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)

                # ControlNet inference process
                control_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)
                down_block_res_samples, mid_block_res_sample = pipe.controlnet(
                    control_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds_fact2,  # Using low-resolution prompt embeddings
                    controlnet_cond=control_image_tensor,         # Control image as condition
                    conditioning_scale=0.5,                         # ControlNet conditioning scale
                    return_dict=False
                )

                noise_pred = pipe.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds_fact1,
                    return_dict=False,
                )[0]
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + 9.0 * (noise_pred_text - noise_pred_uncond)

                noise_pred_fact1 = view.FHigh(noise_pred)

                noise_pred = pipe.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds_fact2,
                    down_block_additional_residuals=down_block_res_samples,  # Additional residuals from down blocks
                    mid_block_additional_residual=mid_block_res_sample,          # Additional residual from mid block
                    return_dict=False,
                )[0]
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred_fact2 = noise_pred_uncond + 7.0 * (noise_pred_text - noise_pred_uncond)
                noise_pred_fact2 = view.FLow(noise_pred_fact2)

                noise_pred = 1.0 * noise_pred_fact1 + 1.0 * noise_pred_fact2
                latents = pipe.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

            # Decode the final latent representation into an image
            image = pipe.vae.decode(latents / pipe.vae.config.scaling_factor, return_dict=False, generator=None)[0]
            do_denormalize = [True] * image.shape[0]
            image = pipe.image_processor.postprocess(image, output_type='pil', do_denormalize=do_denormalize)[0]

        # Save the generated image
        output_path = os.path.join(args.output_dir, f"output_prompt_{j:02d}.jpg")
        image.save(output_path)
        print(f"Image for prompt {j} generated and saved at {output_path}.")

    print("Processing complete.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Stable Diffusion ControlNet with Softedge Generation Script")
    parser.add_argument(
        "--prompt_list",
        type=str,
        nargs='+',
        default=["A photo of a vase with flowers", "A photo of a delicious plate of food"],
        help="High resolution prompt list"
    )
    parser.add_argument(
        "--prompt2",
        type=str,
        default="A photo of a dog",
        help="Low resolution prompt"
    )
    parser.add_argument(
        "--input_image_path",
        type=str,
        default="input_image/1.jpg",
        help="Source image path"
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default="runwayml/stable-diffusion-v1-5",
        help="Model address"
    )
    parser.add_argument(
        "--annotators_model_path",
        type=str,
        default="Annotators",
        help="generating softedge maps"
    )
    parser.add_argument(
        "--model_control_softedge_path",
        type=str,
        default="lllyasviel/control_v11p_sd15_softedge",
        help="ControlNet model address (softedge)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output_ras_unlabeled_attack",
        help="Directory to store generated images"
    )
    parser.add_argument(
        "--img_resolution",
        type=int,
        default=512,
        help="Generated image resolution (both width and height)"
    )
    args = parser.parse_args()
    main(args)

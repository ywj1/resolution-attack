import os
import time
import argparse
from typing import Union, Tuple, Optional

import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms as tvt
from torchvision.transforms import InterpolationMode, GaussianBlur

from diffusers import StableDiffusionPipeline, DDIMInverseScheduler, DDIMScheduler, AutoencoderKL
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import retrieve_timesteps


device = torch.device("cuda:0")


class GaussianBlur2F:
    def __init__(self, kernel_size: int = 3, sigma: float = 21.0):
        self.GB = GaussianBlur(kernel_size, sigma)

    def FHigh(self, noise: torch.Tensor) -> torch.Tensor:
        return noise - self.GB(noise)

    def FLow(self, noise: torch.Tensor) -> torch.Tensor:
        return self.GB(noise)


def load_image(imgname: str, target_size: Optional[Union[int, Tuple[int, int]]] = None) -> torch.Tensor:
    """Load an image and resize it to the target size if provided."""
    pil_img = Image.open(imgname).convert('RGB')
    if target_size is not None:
        if isinstance(target_size, int):
            target_size = (target_size, target_size)
        pil_img = pil_img.resize(target_size, Image.Resampling.LANCZOS)
    return tvt.ToTensor()(pil_img)[None, ...]  # add batch dimension


def img_to_latents(x: torch.Tensor, vae: AutoencoderKL) -> torch.Tensor:
    """Convert an image tensor to latent representation using the VAE."""
    x = 2. * x - 1.
    posterior = vae.encode(x).latent_dist
    latents = posterior.mean * 0.18215
    return latents


def visualize_latents(latents: torch.Tensor):
    """Visualize each channel of the latent representation."""
    latents_np = latents.detach().cpu().numpy()
    num_channels = latents_np.shape[1]
    fig, axes = plt.subplots(1, num_channels, figsize=(20, 4))
    for i in range(num_channels):
        axes[i].imshow(latents_np[0, i], cmap='gray')
        axes[i].axis('off')
        axes[i].set_title(f'Channel {i}')
    plt.show()


@torch.no_grad()
def ddim_inversion(imgname: str, num_steps: int = 50, prompt: str = "", verify: Optional[bool] = False, model_id: str = "") -> torch.Tensor:
    """Perform DDIM inversion on an input image."""
    dtype = torch.float16
    inverse_scheduler = DDIMInverseScheduler.from_pretrained(model_id, subfolder='scheduler')
    pipe = StableDiffusionPipeline.from_pretrained(model_id,
                                                   scheduler=inverse_scheduler,
                                                   safety_checker=None,
                                                   torch_dtype=dtype)
    pipe.to(device)
    vae = pipe.vae

    input_img = load_image(imgname, target_size=(512, 512)).to(device=device, dtype=dtype)
    latents = img_to_latents(input_img, vae)

    inv_latents, _ = pipe(prompt=prompt, negative_prompt="", guidance_scale=1.0,
                          width=input_img.shape[-1], height=input_img.shape[-2],
                          output_type='latent', return_dict=False,
                          num_inference_steps=num_steps, latents=latents)

    if verify:
        pipe.scheduler = DDIMScheduler.from_pretrained(model_id, subfolder='scheduler')
        image = pipe(prompt="", negative_prompt="", guidance_scale=1.0,
                     num_inference_steps=num_steps, latents=inv_latents)
        fig, ax = plt.subplots(1, 2)
        ax[0].imshow(tvt.ToPILImage()(input_img[0]))
        ax[1].imshow(image.images[0])
        plt.show()
    return inv_latents


def main(args):
    # Initialize the Stable Diffusion pipeline with the specified model
    pipe = StableDiffusionPipeline.from_pretrained(args.model_id, torch_dtype=torch.float16)
    pipe = pipe.to(device)
    negative_prompt = "not centered, blurry, unrealistic, distorted"

    # Create an instance of the GaussianBlur2F class
    view = GaussianBlur2F()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Loop over a fixed number of images (here, i ranges from 1 to 1)
    for i in range(1, 2):
        start_time = time.time()

        # Loop over each high-resolution prompt in the list
        for j, prompt1 in enumerate(args.prompt_list):
            prompt_embeds_fact1, neg_prompt_embeds_fact1 = pipe.encode_prompt(
                prompt1, device=device, num_images_per_prompt=1, do_classifier_free_guidance=True
            )
            prompt_embeds_fact2, neg_prompt_embeds_fact2 = pipe.encode_prompt(
                args.prompt2, device=device, num_images_per_prompt=1, do_classifier_free_guidance=True
            )

            # Encode negative prompt
            neg_prompt_embeds_fact, _ = pipe.encode_prompt(
                negative_prompt, device=device, num_images_per_prompt=1, do_classifier_free_guidance=True
            )

            # Combine positive and negative prompt embeddings for CFG
            prompt_embeds_fact1 = torch.cat([neg_prompt_embeds_fact, prompt_embeds_fact1])
            prompt_embeds_fact2 = torch.cat([neg_prompt_embeds_fact, prompt_embeds_fact2])
            num_channels_latents = pipe.unet.config.in_channels
            latents = pipe.prepare_latents(
                1, num_channels_latents, args.img_resolution, args.img_resolution,
                prompt_embeds_fact1.dtype, device, None, None
            )

            # Retrieve timesteps for the scheduler
            timesteps, _ = retrieve_timesteps(pipe.scheduler, 300, device, None)

            with torch.no_grad():
                total_timesteps = len(timesteps)
                one_eighth_index = total_timesteps // 15
                seven_eighths_index = 14 * total_timesteps // 15

                for idx, t in enumerate(timesteps):
                    latent_model_input = torch.cat([latents] * 2)
                    latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)

                    if idx < one_eighth_index:
                        noise_pred = pipe.unet(
                            latent_model_input, t,
                            encoder_hidden_states=prompt_embeds_fact2,
                            timestep_cond=None, cross_attention_kwargs=None,
                            added_cond_kwargs=None, return_dict=False,
                        )[0]
                        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                        noise_pred = noise_pred_uncond + 7.0 * (noise_pred_text - noise_pred_uncond)

                    elif one_eighth_index <= idx < seven_eighths_index:
                        noise_pred = pipe.unet(
                            latent_model_input, t,
                            encoder_hidden_states=prompt_embeds_fact1,
                            timestep_cond=None, cross_attention_kwargs=None,
                            added_cond_kwargs=None, return_dict=False,
                        )[0]
                        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                        noise_pred = noise_pred_uncond + 9.0 * (noise_pred_text - noise_pred_uncond)

                        noise_pred_fact1 = view.FHigh(noise_pred)

                        noise_pred = pipe.unet(
                            latent_model_input, t,
                            encoder_hidden_states=prompt_embeds_fact2,
                            timestep_cond=None, cross_attention_kwargs=None,
                            added_cond_kwargs=None, return_dict=False,
                        )[0]
                        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                        noise_pred_fact2 = noise_pred_uncond + 7.0 * (noise_pred_text - noise_pred_uncond)
                        noise_pred_fact2 = view.FLow(noise_pred_fact2)

                        noise_pred = 1.0 * noise_pred_fact1 + 1.0 * noise_pred_fact2
                    else:
                        noise_pred = pipe.unet(
                            latent_model_input, t,
                            encoder_hidden_states=prompt_embeds_fact1,
                            timestep_cond=None, cross_attention_kwargs=None,
                            added_cond_kwargs=None, return_dict=False,
                        )[0]
                        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                        noise_pred = noise_pred_uncond + 9.0 * (noise_pred_text - noise_pred_uncond)

                    latents = pipe.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

                # Decode the final latent representation into an image
                image = pipe.vae.decode(latents / pipe.vae.config.scaling_factor, return_dict=False, generator=None)[0]
                do_denormalize = [True] * image.shape[0]
                image = pipe.image_processor.postprocess(image, output_type='pil', do_denormalize=do_denormalize)[0]

            # Save the generated image
            output_path = os.path.join(args.output_dir, f"output_{i:02d}_prompt_{j:02d}.jpg")
            image.save(output_path)
            print(f"Image {i} for prompt {j} generated and saved at {output_path}.")

        end_time = time.time()
        print(f"Time taken for image {i}: {end_time - start_time:.2f} seconds")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Stable Diffusion Generation Script")
    parser.add_argument(
        "--prompt_list",
        type=str,
        nargs='+',
        default=["A photo of a delicious plate of food", "A photo of a lion's head"],
        help="High resolution prompt list"
    )
    parser.add_argument(
        "--prompt2",
        type=str,
        default="A photo of a dog's head",
        help="Low resolution prompt"
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default="sd-legacy/stable-diffusion-v1-5",
        help="Model address"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/output_ra_attack",
        help="Directory to store images"
    )
    parser.add_argument(
        "--img_resolution",
        type=int,
        default=512,
        help="Generated image resolution (both width and height)"
    )
    args = parser.parse_args()
    main(args)

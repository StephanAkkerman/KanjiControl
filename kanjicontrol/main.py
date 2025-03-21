import torch
from diffusers import (
    AutoencoderKL,
    ControlNetModel,
    DDIMScheduler,
    StableDiffusionControlNetPipeline,
)
from diffusers.utils import load_image
from PIL import Image

# https://huggingface.co/spaces/AP123/IllusionDiffusion/blob/main/app.py
vae = AutoencoderKL.from_pretrained(
    "stabilityai/sd-vae-ft-mse", torch_dtype=torch.float16
)
controlnet = ControlNetModel.from_pretrained(
    "monster-labs/control_v1p_sd15_qrcode_monster",
    torch_dtype=torch.bfloat16,
    cache_dir="models",
)

pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    controlnet=controlnet,
    # vae=vae,
    safety_checker=None,
    torch_dtype=torch.bfloat16,
    cache_dir="models",
).to("cuda")

pipe.enable_xformers_memory_efficient_attention()
pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
pipe.enable_model_cpu_offload()


def resize_for_condition_image(input_image: Image, resolution: int):
    input_image = input_image.convert("RGB")
    W, H = input_image.size
    k = float(resolution) / min(H, W)
    H *= k
    W *= k
    H = int(round(H / 64.0)) * 64
    W = int(round(W / 64.0)) * 64
    img = input_image.resize((W, H), resample=Image.LANCZOS)
    return img


# play with guidance_scale, controlnet_conditioning_scale and strength to make a valid QR Code Image

# qr code image
source_image = load_image("./output/日.png")
condition_image = resize_for_condition_image(source_image, 768)
generator = torch.manual_seed(123121231)
image = pipe(
    prompt="sun",
    negative_prompt="ugly, disfigured, low quality, blurry, nsfw",
    image=condition_image,
    width=768,
    height=768,
    guidance_scale=20,
    controlnet_conditioning_scale=1.5,
    generator=generator,
    strength=0.9,
    num_inference_steps=15,
)

image.images[0].save("output/qr_code.png")

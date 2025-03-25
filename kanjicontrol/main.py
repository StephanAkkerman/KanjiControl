import torch
from char2img import get_char_img
from diffusers import (
    AutoencoderKL,
    ControlNetModel,
    DDIMScheduler,
    StableDiffusionControlNetPipeline,
)
from PIL import Image


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


class KanjiControl:
    def __init__(self):

        # https://huggingface.co/spaces/AP123/IllusionDiffusion
        # https://huggingface.co/spaces/AP123/IllusionDiffusion/blob/main/app.py
        vae = AutoencoderKL.from_pretrained(
            "stabilityai/sd-vae-ft-mse", torch_dtype=torch.float16
        )

        # SD 1.5: https://huggingface.co/monster-labs/control_v1p_sd15_qrcode_monster
        # https://huggingface.co/DionTimmer/controlnet_qrcode-control_v1p_sd15
        controlnet = ControlNetModel.from_pretrained(
            "monster-labs/control_v1p_sd15_qrcode_monster",
            # subfolder="v2",
            torch_dtype=torch.float16,
            cache_dir="models",
        )

        # "runwayml/stable-diffusion-v1-5"
        # "SG161222/Realistic_Vision_V5.1_noVAE"
        # https://huggingface.co/darkstorm2150/Protogen_x5.8_Official_Release (some errors)
        # https://huggingface.co/digiplay/Photon_v1 (quite good)
        # https://huggingface.co/RunDiffusion/Juggernaut-XL-v9 (doesnt work)
        # https://huggingface.co/Lykon/dreamshaper-8 (not great)
        # https://huggingface.co/stablediffusionapi/deliberate-v3
        self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
            "digiplay/Photon_v1",
            # vae=vae,
            controlnet=controlnet,
            torch_dtype=torch.float16,
            cache_dir="models",
        ).to("cuda")

        self.pipe.enable_xformers_memory_efficient_attention()
        self.pipe.scheduler = DDIMScheduler.from_config(self.pipe.scheduler.config)

    def create_image(self, char: str, meaning: str):
        # Create the image
        source_img = get_char_img(char, "data/thin.ttf")

        # Get the meaning

        # Ask LLM to create a prompt for the image model

        # condition_image = resize_for_condition_image(source_image, 512)
        image = self.pipe(
            prompt=meaning,
            negative_prompt="low quality",
            image=source_img,
            width=512,
            height=512,
            guidance_scale=12,  # 0 - 50
            controlnet_conditioning_scale=1.5,  # "Illusion strength" 0-5 (1.5 is default)
            control_guidance_start=0,
            control_guidance_end=1,
            generator=torch.manual_seed(123121231),
            strength=0.9,
            num_inference_steps=20,
        )

        image.images[0].save(f"output/{char}.png")


if __name__ == "__main__":
    kc = KanjiControl()
    kc.create_image("火", "fire")

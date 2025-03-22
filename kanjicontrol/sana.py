import torch
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"

# Wait for diffusers release: https://github.com/NVlabs/Sana/blob/main/asset/docs/model_zoo.md

pipe = SanaControlNetPipeline(
    "configs/sana_controlnet_config/Sana_600M_img1024_controlnet.yaml"
)
pipe.from_pretrained(
    "hf://Efficient-Large-Model/Sana_600M_1024px_ControlNet_HED/checkpoints/Sana_600M_1024px_ControlNet_HED.pth"
)

ref_image = Image.open(
    "asset/controlnet/ref_images/A transparent sculpture of a duck made out of glass. The sculpture is in front of a painting of a la.jpg"
)
prompt = "A transparent sculpture of a duck made out of glass. The sculpture is in front of a painting of a landscape."

images = pipe(
    prompt=prompt,
    ref_image=ref_image,
    guidance_scale=4.5,
    num_inference_steps=10,
    sketch_thickness=2,
    generator=torch.Generator(device=device).manual_seed(0),
)

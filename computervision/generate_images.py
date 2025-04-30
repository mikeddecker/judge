from diffusers import StableDiffusionPipeline
import torch

pipe = StableDiffusionPipeline.from_pretrained(
    "Aungkhine/Simbolo_Text_to_Image_Generator_V3",
    torch_dtype=torch.float16,
)

pipe.to("cuda")
pipe.enable_xformers_memory_efficient_attention()

# Generate 512x512 normally
prompt = """
MIKE as a logo, but the letters of Mike a jump rope.
"""

prompt="a creative jump rope in pvc"

rounds = 12
multiplier = 6

for r in range(rounds):
    DIM = 512
    image = pipe(
        prompt=prompt,
        height=DIM,
        width=DIM,
        num_inference_steps=(2 + r) * multiplier,
        guidance_scale=7.5
    ).images[0]

    image.save(f"simbolo_output_{r}.png")

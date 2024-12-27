import torch
from diffnext.pipelines import NOVAPipeline
from diffnext.utils import export_to_image, export_to_video

model_id = "BAAI/nova-d48w1024-osp480"
model_args = {"torch_dtype": torch.float16, "trust_remote_code": True}
pipe = NOVAPipeline.from_pretrained(model_id, **model_args)
pipe = pipe.to("cuda")

prompt = "Many spotted jellyfish pulsating under water."

image = pipe(prompt, max_latent_length=1).frames[0, 0]
export_to_image(image, "jellyfish.jpg")

video = pipe(prompt, max_latent_length=9).frames[0]
export_to_video(video, "jellyfish.mp4", fps=12)

# Increase AR and diffusion steps for better video quality.
video = pipe(
    prompt,
    max_latent_length=9,
    num_inference_steps=128,  # default: 64
    num_diffusion_steps=100,  # default: 25
).frames[0]
export_to_video(video, "jellyfish_v2.mp4", fps=12)
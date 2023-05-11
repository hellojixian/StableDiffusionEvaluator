import torch
import math
import random
import shutil

from diffusers import StableDiffusionPipeline
from libs.clip_evaluator import ClipEvaluator
from libs.googlesearch import SearchImage

seed = 1234
random.seed(seed)

model_path = 'runwayml/stable-diffusion-v1-5'
height = 512                        # default height of Stable Diffusion
width = 512                         # default width of Stable Diffusion
num_inference_steps = 40            # Number of denoising steps
guidance_scale = 7.5                # Scale for classifier-free guidance
generator = torch.manual_seed(random.random() * 2 ** 32)   # Seed generator to create the inital latent noise
output_filename = f"output/generate_image.png"

# init clip evaluator
ce = ClipEvaluator()

prompt = "a cat is running on the grass"
print(f"prompt score: {ce.evaluate_prompt(prompt)}")

# generate an image with prompt
pipe = StableDiffusionPipeline.from_pretrained(model_path, torch_dtype=torch.float16)
pipe.to("cuda" if torch.cuda.is_available() else "cpu")
image = pipe(prompt,
              width=width,
              height=height,
              num_inference_steps=num_inference_steps,
              guidance_scale=guidance_scale,
              generator=generator,
              ).images[0]
image.save(output_filename)
genenerated_image_score = ce.evaluate_prompt_image(prompt, output_filename)
print(f"generate image score: {genenerated_image_score}")

# fetch a ground truth image from google search
reference_image = random.choice(SearchImage(prompt, num_images=5))
reference_image_score = ce.evaluate_prompt_image(prompt, reference_image.path)
shutil.move(reference_image.path, "output/reference_image.png")
print(f"grand true image score: {reference_image_score}")

overall_score = math.exp(genenerated_image_score - reference_image_score) - 1
print(f"overall score: {overall_score}")

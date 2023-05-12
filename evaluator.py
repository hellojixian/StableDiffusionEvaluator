import torch
import sys, os
import random
import shutil
import logging

from diffusers import StableDiffusionPipeline
from libs.clip_evaluator import ClipEvaluator
from libs.googlesearch import SearchImage

logging.disable(logging.WARNING)
seed = 1234
random.seed(seed)

model_path = 'runwayml/stable-diffusion-v1-5'
height = 512                        # default height of Stable Diffusion
width = 512                         # default width of Stable Diffusion
num_inference_steps = 40            # Number of denoising steps
guidance_scale = 7.5                # Scale for classifier-free guidance
generator = torch.manual_seed(random.random() * 2 ** 32)   # Seed generator to create the inital latent noise
output_filename = f"output/generate_image.png"

gray_escape = "\033[90m"
reset_escape = "\033[0m"

# init clip evaluator
ce = ClipEvaluator()

prompt = " ".join(sys.argv[1:]) or "a photograph of an astronaut riding a horse"
print(f"prompt score:\t\t\t {ce.evaluate_prompt(prompt)}\n\
{gray_escape}value higher would be better, \nindicate how meaningful the prompt.\n{reset_escape}")

# generate an image with prompt
pipe = StableDiffusionPipeline.from_pretrained(model_path, torch_dtype=torch.float16)
pipe.set_progress_bar_config(disable=True)
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
print(f"generate image score:\t\t {genenerated_image_score}\n\
{gray_escape}value higher would be better, \nthe distance regarding the generated image to prompt.\n{reset_escape}")

# fetch a ground truth image from google search
reference_images = SearchImage(prompt, num_images=5)
reference_image = random.choice(reference_images)
reference_image_score = ce.evaluate_prompt_image(prompt, reference_image.path)
shutil.copy(reference_image.path, "output/reference_image.png")
print(f"ground true image score:\t {reference_image_score}\n\
{gray_escape}value higher would be better, \nthe distance regarding the generated image to prompt and ground truth image to prompt.\n{reset_escape}")

overall_score = (genenerated_image_score - reference_image_score)
print(f"overall score:\t\t\t {overall_score}\n\
{gray_escape}positive and higher value would be better\n{reset_escape}")

# compare generated image to ground true image semantic difference
image_loss = ce.evaluate_image_images(output_filename, [img.path for img in reference_images])
print(f"image to ground true loss:\t {image_loss} \n\
{gray_escape}value lower would be better\n{reset_escape}")

# clean up
for image in reference_images: os.remove(image.path)


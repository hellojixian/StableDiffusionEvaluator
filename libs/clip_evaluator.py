import torch
import math
from transformers import CLIPProcessor, CLIPModel
from PIL import Image

clip_model_path = "openai/clip-vit-large-patch14"

class ClipEvaluator:
  def __init__(self):
    self.device =  "cuda" if torch.cuda.is_available() else "cpu"
    self.model = CLIPModel.from_pretrained(clip_model_path).to(self.device)
    self.processor = CLIPProcessor.from_pretrained(clip_model_path)
    self.tokenizer = self.processor.tokenizer
    self.random_features = self._generate_random_features()
    pass


  @torch.no_grad()
  def _generate_random_features(self):
    random_prompts = ["", "sks", "asfasdf"]
    random_features = torch.zeros((0, self.model.config.projection_dim)).to(self.device)
    with torch.no_grad():
      for prompt in random_prompts:
        random_features = torch.cat([random_features, self.model.get_text_features(
          **self.tokenizer(prompt, return_tensors="pt", padding=True).to(self.device))])
    return random_features

  @torch.no_grad()
  def evaluate_prompt(self, prompt):
    test_features = self.model.get_text_features(
      **self.tokenizer(prompt, return_tensors="pt", padding=True).to(self.device))

    similarity = torch.cosine_similarity(test_features, self.random_features, dim=1)
    score = (1 - torch.mean(similarity).cpu().detach().numpy())
    return score

  @torch.no_grad()
  def evaluate_prompt_image(self, prompt, image):
    test_features = self.model.get_text_features(
      **self.tokenizer(prompt, return_tensors="pt", padding=True).to(self.device))

    image_features = self.model.get_image_features(
      **self.processor(images=Image.open(image), return_tensors="pt").to(self.device))

    random_similarity = torch.cosine_similarity(image_features, self.random_features, dim=1)
    random_similarity = torch.mean(random_similarity).cpu().detach().numpy()
    prompt_similarity = torch.cosine_similarity(image_features, test_features, dim=1).cpu().detach().numpy()[0]
    # print(random_similarity, prompt_similarity)
    return math.exp(prompt_similarity - random_similarity)

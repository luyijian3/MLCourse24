import os
import json
from diffusers import StableDiffusionPipeline
import torch
import random
from PIL import Image
from torchvision import transforms
#from face_recog_v4 import load_face_recog_model, face_recog
from utils import face_recog
from face_recog_v4 import load_face_recog_model, face_recog_v4
import numpy as np
import pickle
import time
from sklearn.preprocessing import LabelEncoder
import torch.nn as nn



# Generate image using Stable Diffusion
def generate_image(pipe, prompt, save_path):
    try:
        image = pipe(prompt).images[0]
        image.save(save_path)
        print(f"Image saved temporarily to: {save_path}")
        return save_path
    except Exception as e:
        print(f"Error generating image for prompt: {prompt}\nError: {e}")
        return None


# Gemini class for generating prompts
class Gemini:
    def __init__(self, model="gemini-1.5-flash"):
        api_key = os.environ["gemini_API_KEY"]
        import google.generativeai as genai
        genai.configure(api_key=api_key, transport="rest")
        self.model = genai.GenerativeModel(model_name=model)

    def get_prompts(self, name):
        prompt = f"Write 5 implicit prompts that clearly describe the following celebrity: {name}, without mentioning their name. Elements like gender, country, career, and representative achievements should be included."
        try:
            response = self.model.generate_content([prompt])
            time.sleep(5)
            response = response.text.strip().split("\n")
            return [i for i in response if i.replace(' ','')!=''][:3]
        except Exception as e:
            print(f"Error generating prompts for {name}: {e}")
            return []


# Main workflow
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    v4_model, label_encoder = load_face_recog_model()
    # Load pre-trained model and label encoder
    #model, label_encoder = load_face_recog_model()
    #model.eval()
    pipe = StableDiffusionPipeline.from_pretrained("sd-legacy/stable-diffusion-v1-5")
    pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")
    # Load prompts from JSONL file
    input_file = "Celebrity_Prompts.jsonl"
    if not os.path.exists(input_file):
        print(f"Error: {input_file} does not exist. Please generate prompts first.")
        exit()

    with open(input_file, "r", encoding="utf-8") as jsonl_file:
        generated_prompts = [json.loads(line) for line in jsonl_file]

    # Create output folder
    output_folder = "Generated_Images"
    os.makedirs(output_folder, exist_ok=True)

    # File to save successful results
    results_file = "filtered_results_0.jsonl"

    # Initialize Gemini for generating new prompts
    gemini = Gemini()
    chance = 5
    # Iterate through all celebrities in the JSONL file
    for entry in generated_prompts[14:]:
        chance = 5
        name = entry["name"]
        category = entry["category"]

        successful_prompts = []
        while len(successful_prompts) < 1:  # Ensure 5 valid prompts per celebrity
            chance -= 1
            prompts = gemini.get_prompts(name)
            for prompt in prompts:
                successful_images = []
                for i in range(5):  # Generate 5 images per prompt
                    print(f"Generating image {i + 1}/5 for {name} ({category}) with prompt: {prompt}")

                    file_name = f"{name.replace(' ', '_')}_{'_'.join(prompt.split()[:3])}_{i + 1}.png"
                    save_path = os.path.join(output_folder, file_name)

                    image_path = generate_image(pipe, prompt, save_path)
                    if image_path is None:
                        continue

                    confidence = face_recog(image_path, category, name.replace(' ','_'))
                    _, confidence_v4 = face_recog_v4(image_path, v4_model, label_encoder)
                    #print(detected_name,'|',name)
                    #if detected_name == name and confidence >= 0.6:
                    if confidence and confidence_v4>=0:
                        successful_images.append(image_path)
                    else:
                        os.remove(image_path)

                if len(successful_images) >= 3:  # Check if 3/5 images are valid
                    print(f"Prompt validated for {name}: {prompt}")
                    successful_prompts.append({"prompt": prompt, "images": successful_images})

                if len(successful_prompts) >= 1:
                    break
            if chance <= 0:
                break

        # Save successful prompts and images
        with open(results_file, "a", encoding="utf-8") as results:
            for prompt_entry in successful_prompts:
                json.dump(
                    {"name": name, "category": category, "prompt": prompt_entry["prompt"], "images": prompt_entry["images"]},
                    results,
                    ensure_ascii=False,
                )
                results.write("\n")

    print("Image generation process complete.")

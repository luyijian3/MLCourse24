import os
import json
from diffusers import StableDiffusionPipeline
import torch
import random
from PIL import Image
from torchvision import transforms
import face_recognition
import numpy as np
import pickle
import time
from sklearn.preprocessing import LabelEncoder
import torch.nn as nn


# Extract face from the image
def extract_face(image_path, required_size=(160, 160)):
    try:
        image = face_recognition.load_image_file(image_path)
        face_locations = face_recognition.face_locations(image)
        if not face_locations:
            return None
        top, right, bottom, left = face_locations[0]
        face_image = image[top:bottom, left:right]
        pil_image = Image.fromarray(face_image)
        pil_image = pil_image.resize(required_size)
        return pil_image
    except Exception as e:
        print(f"Error processing image {image_path}: {str(e)}")
        return None


# Define EfficientNet-based model
class EfficientNetClassifier(nn.Module):
    def __init__(self, num_classes):
        super(EfficientNetClassifier, self).__init__()
        from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
        self.efficientnet = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        self.efficientnet.classifier[1] = nn.Linear(self.efficientnet.classifier[1].in_features, num_classes)

    def forward(self, x):
        return self.efficientnet(x)
    
    
    


# Load the label encoder
def load_label_encoder(filename='label_encoder.pkl'):
    with open(filename, 'rb') as f:
        return pickle.load(f)


# Recognize face using the pre-trained EfficientNet model
def face_recog(image_path, model=None, label_encoder=None, device='cuda', confidence_threshold=0.6):
    """
    Perform local face recognition using a pre-trained model.

    Args:
        image_path (str): Path to the image to be recognized.
        model (nn.Module): Pre-trained EfficientNet model.
        label_encoder (LabelEncoder): Encoder to map class indices to celebrity names.
        device (str): Device to use for inference ('cuda' or 'cpu').
        confidence_threshold (float): Minimum confidence to consider a recognition successful.

    Returns:
        str: Predicted celebrity name or error message.
        float: Confidence score for the prediction.
    """
    try:
        face_image = extract_face(image_path)
        if face_image is None:
            return "No Face Detected", 0.0

        transform = transforms.Compose([
            transforms.Resize((160, 160)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        face_tensor = transform(face_image).unsqueeze(0).to(device)

        model.eval()
        with torch.no_grad():
            outputs = model(face_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted_class = torch.max(probabilities, 1)

        if confidence.item() < confidence_threshold:
            return "Low Confidence", confidence.item()

        predicted_label = label_encoder.inverse_transform([predicted_class.item()])[0]
        return predicted_label, confidence.item()
    except Exception as e:
        print(f"Error recognizing face: {str(e)}")
        return "Recognition Error", 0.0


# Stable Diffusion pipeline setup
pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")


# Generate image using Stable Diffusion
def generate_image(prompt, save_path):
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
        api_key = os.environ["GEMINI_API_KEY"]
        import google.generativeai as genai
        genai.configure(api_key=api_key, transport="rest")
        self.model = genai.GenerativeModel(model_name=model)

    def get_prompts(self, name):
        prompt = f"Write 5 implicit prompts that clearly describe the following celebrity: {name}, without mentioning their name. Elements like gender, country, career, and representative achievements should be included."
        try:
            response = self.model.generate_content([prompt])
            time.sleep(5)
            return response.text.strip().split("\n")
        except Exception as e:
            print(f"Error generating prompts for {name}: {e}")
            return []


# Main workflow
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load pre-trained model and label encoder
    label_encoder = load_label_encoder('label_encoder.pkl')
    num_classes = len(label_encoder.classes_)
    model = EfficientNetClassifier(num_classes=num_classes).to(device)
    model.load_state_dict(torch.load("best_model.pth"))
    model.eval()

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
    results_file = "successful_results.jsonl"

    # Initialize Gemini for generating new prompts
    gemini = Gemini()

    # Iterate through all celebrities in the JSONL file
    for entry in generated_prompts:
        name = entry["name"]
        category = entry["category"]

        successful_prompts = []
        while len(successful_prompts) < 5:  # Ensure 5 valid prompts per celebrity
            prompts = gemini.get_prompts(name)
            for prompt in prompts:
                successful_images = []
                for i in range(5):  # Generate 5 images per prompt
                    print(f"Generating image {i + 1}/5 for {name} ({category}) with prompt: {prompt}")

                    file_name = f"{name.replace(' ', '_')}_{'_'.join(prompt.split()[:3])}_{i + 1}.png"
                    save_path = os.path.join(output_folder, file_name)

                    image_path = generate_image(prompt, save_path)
                    if image_path is None:
                        continue

                    detected_name, confidence = face_recog(image_path, model, label_encoder, device)
                    if detected_name == name and confidence >= 0.6:
                        successful_images.append(image_path)
                    else:
                        os.remove(image_path)

                if len(successful_images) >= 3:  # Check if 3/5 images are valid
                    print(f"Prompt validated for {name}: {prompt}")
                    successful_prompts.append({"prompt": prompt, "images": successful_images})

                if len(successful_prompts) >= 5:
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

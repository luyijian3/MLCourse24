import os
import json
from diffusers import StableDiffusionPipeline
import torch
from PIL import Image
import time
import face_recognition

# Load the Stable Diffusion model from Hugging Face
pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")  # Use GPU if available

# Function to generate an image using Stable Diffusion
def generate_image(prompt, save_path):
    try:
        image = pipe(prompt).images[0]  # Generate an image from the prompt
        image.save(save_path)  # Save the image to the specified path
        print(f"Image saved temporarily to: {save_path}")
        return save_path
    except Exception as e:
        print(f"Error generating image for prompt: {prompt}\nError: {e}")
        return None

# Function to verify if the generated image matches the target celebrity using face recognition
def is_correct_celebrity(image_path, celebrity_name):
    """
    等田老师写好再加进来，先放一个在这里
    """
    try:
        # Path to the folder containing reference images
        reference_folder = "Celebrity_Image_Dataset"
        reference_image_path = os.path.join(reference_folder, f"{celebrity_name.replace(' ', '_')}.jpg")

        if not os.path.exists(reference_image_path):
            print(f"Reference image for {celebrity_name} not found: {reference_image_path}")
            return False

        # Load the reference image and extract its face encoding
        reference_image = face_recognition.load_image_file(reference_image_path)
        reference_encodings = face_recognition.face_encodings(reference_image)
        if len(reference_encodings) == 0:
            print(f"No face found in reference image for {celebrity_name}")
            return False

        reference_encoding = reference_encodings[0]

        # Load the generated image and extract its face encoding
        generated_image = face_recognition.load_image_file(image_path)
        generated_encodings = face_recognition.face_encodings(generated_image)
        if len(generated_encodings) == 0:
            print(f"No face found in generated image: {image_path}")
            return False

        # Compare the generated image with the reference image
        results = face_recognition.compare_faces([reference_encoding], generated_encodings[0], tolerance=0.6)
        return results[0]  # Return True if the faces match
    except Exception as e:
        print(f"Error in face recognition for {celebrity_name}: {e}")
        return False

class Gemini:
    def __init__(self, model="gemini-1.5-flash"):
        api_key = os.environ["GEMINI_API_KEY"]
        import google.generativeai as genai
        genai.configure(api_key=api_key, transport="rest")
        self.model = genai.GenerativeModel(model_name=model)

    def get_prompt(self, name):
        """
        Generate a single implicit prompt describing the given celebrity.
        """
        prompt = f"Write 1 implicit prompts that clearly describe the following celebrity: {name}, without mentioning their name. Elements like gender, country, career, representative achievements should be included in each prompt. Take Taylor Swift as example, a possible prompt is: American female singer known for her country-pop hits and songwriter, like \"Love Story\" and \"Shake It Off\". Other irrelevant sentences should not be inclued in your answer."
        try:
            response = self.model.generate_content([prompt])
            time.sleep(5)
            return response.text.strip()  # Return the generated prompt as a string
        except Exception as e:
            print(f"Error generating prompt for {name}: {e}")
            return None

if __name__ == "__main__":

    input_file = "Celebrity_Prompts.jsonl"
    if not os.path.exists(input_file):
        print(f"Error: {input_file} does not exist. Please generate prompts first.")
        exit()

    with open(input_file, "r", encoding="utf-8") as jsonl_file:
        generated_prompts = [json.loads(line) for line in jsonl_file]

    output_folder = "Generated_Images"
    os.makedirs(output_folder, exist_ok=True)

    # File to save the successful results (images and prompts)
    results_file = "successful_results.jsonl"

    gemini = Gemini()

    # Iteration
    for entry in generated_prompts:
        name = entry["name"]
        category = entry["category"]
        prompts = entry["prompts"]

        for prompt in prompts:
            while True:  # Loop until a successful match is found
                print(f"Generating image for {name} ({category}) with prompt: {prompt}")

                # Define the file name and path for the generated image
                file_name = f"{name.replace(' ', '_')}_{'_'.join(prompt.split()[:3])}.png"
                save_path = os.path.join(output_folder, file_name)

                # Generate the image
                image_path = generate_image(prompt, save_path)
                if image_path is None:
                    print(f"Failed to generate image for prompt: {prompt}")
                    break

                # Perform face recognition to verify the image
                if is_correct_celebrity(image_path, name):
                    print(f"Image recognized as {name}. Saving prompt and image.")

                    # Save the successful prompt and image path to the results file
                    with open(results_file, "a", encoding="utf-8") as results:
                        json.dump({"name": name, "category": category, "prompt": prompt, "image_path": save_path}, results, ensure_ascii=False)
                        results.write("\n")

                    break
                else:
                    print(f"Image not recognized as {name}. Deleting image and generating a new prompt...")
                    os.remove(image_path)  # Delete the unrecognized image

                    # Generate a new prompt
                    new_prompt = gemini.get_prompt(name)
                    if new_prompt:
                        prompt = new_prompt
                    else:
                        print(f"Failed to generate new prompt for {name}. Skipping...")
                        break

    print("Image generation process complete.")

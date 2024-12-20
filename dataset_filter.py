import os
import json
from diffusers import StableDiffusionPipeline
import torch
from PIL import Image
import time
import face_recognition


# Function to load candidate faces from the dataset
def load_candidate_faces(dataset_path):
    """
    Load one face encoding per celebrity from the dataset.

    Args:
        dataset_path (str): Path to the celebrity dataset.

    Returns:
        tuple: A list of candidate face encodings and a list of corresponding names.
    """
    candidate_faces = []
    candidate_names = []

    for category in os.listdir(dataset_path):
        category_path = os.path.join(dataset_path, category)
        if os.path.isdir(category_path):
            for celebrity in os.listdir(category_path):
                celebrity_path = os.path.join(category_path, celebrity)
                if os.path.isdir(celebrity_path):
                    for img_file in os.listdir(celebrity_path):
                        if img_file.endswith(('jpg', 'jpeg', 'png')):
                            img_path = os.path.join(celebrity_path, img_file)
                            image = face_recognition.load_image_file(img_path)
                            encodings = face_recognition.face_encodings(image)
                            if encodings:  
                                candidate_faces.append(encodings[0])  # Use the first encoding
                                candidate_names.append(f"{category}/{celebrity}")  # Save category/celebrity name
                            break  # Select only one image per celebrity

    if not candidate_faces:
        raise ValueError("No valid faces found in the dataset. Please check the dataset structure and image quality.")

    return candidate_faces, candidate_names


# Function to get the recognition score for a generated image
def get_recognition_score(image_path, celebrity_name, candidate_faces, candidate_names, tolerance=0.6):
    """
    Compare the generated image's face encoding with the dataset encodings.

    Args:
        image_path (str): Path to the generated image.
        celebrity_name (str): Name of the target celebrity.
        candidate_faces (list): List of face encodings from the dataset.
        candidate_names (list): List of corresponding names for the face encodings.
        tolerance (float): Tolerance for face recognition (default is 0.6).

    Returns:
        float: Recognition score (1.0 for a perfect match, 0.0 if no match).
    """
    try:
        # Load the test image
        test_image = face_recognition.load_image_file(image_path)
        test_encodings = face_recognition.face_encodings(test_image)

        if not test_encodings:
            print(f"No face detected in the generated image: {image_path}")
            return 0.0  # No face detected

        test_encoding = test_encodings[0]

        # Compare the test image encoding with candidate face encodings
        results = face_recognition.compare_faces(candidate_faces, test_encoding, tolerance)
        face_distances = face_recognition.face_distance(candidate_faces, test_encoding)

        # Check if the test image matches the given celebrity
        if True in results:
            match_index = results.index(True)
            matched_name = candidate_names[match_index]

            # Ensure the matched name corresponds to the target celebrity
            if celebrity_name in matched_name:
                print(f"Matched {celebrity_name} with recognition score: {1 - face_distances[match_index]:.2f}")
                return 1 - face_distances[match_index]  # Return similarity score
            else:
                print(f"Mismatch: Detected {matched_name} instead of {celebrity_name}.")
                return 0.0
        else:
            print(f"No match found for {celebrity_name}.")
            return 0.0  # No match found
    except Exception as e:
        print(f"Error in face recognition for {celebrity_name}: {e}")
        return 0.0


# Stable Diffusion pipeline setup
pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")  # Use GPU if available

# Function to generate an image using Stable Diffusion
def generate_image(prompt, save_path):
    """
    Generate an image from the given prompt using Stable Diffusion.

    Args:
        prompt (str): The text prompt for image generation.
        save_path (str): Path to save the generated image.

    Returns:
        str: Path to the saved image or None if generation fails.
    """
    try:
        image = pipe(prompt).images[0]  # Generate an image from the prompt
        image.save(save_path)  # Save the image to the specified path
        print(f"Image saved temporarily to: {save_path}")
        return save_path
    except Exception as e:
        print(f"Error generating image for prompt: {prompt}\nError: {e}")
        return None


# Main program
if __name__ == "__main__":
    # Path to the dataset and JSONL file
    dataset_path = "Celebrity_Image_Dataset"
    input_file = "Celebrity_Prompts.jsonl"

    # Check if the input JSONL file exists
    if not os.path.exists(input_file):
        print(f"Error: {input_file} does not exist. Please generate prompts first.")
        exit()

    # Load candidate images (one per celebrity)
    print("Loading candidate faces and names from the dataset...")
    candidate_faces, candidate_names = load_candidate_faces(dataset_path)

    # Load prompts from the JSONL file
    with open(input_file, "r", encoding="utf-8") as jsonl_file:
        generated_prompts = [json.loads(line) for line in jsonl_file]

    # Create output folder for generated images
    output_folder = "Generated_Images"
    os.makedirs(output_folder, exist_ok=True)

    # File to save successful results (images and prompts)
    results_file = "successful_results.jsonl"

    # Iterate over all entries in the JSONL file
    for entry in generated_prompts:
        name = entry["name"]
        category = entry["category"]
        prompts = entry["prompts"]

        for prompt in prompts:
            successful_images = []  # To store recognized images and prompts
            for i in range(5):  # Generate 5 images for each prompt
                print(f"Generating image {i + 1}/5 for {name} ({category}) with prompt: {prompt}")

                # Define the file name and path for the generated image
                file_name = f"{name.replace(' ', '_')}_{'_'.join(prompt.split()[:3])}_{i + 1}.png"
                save_path = os.path.join(output_folder, file_name)

                # Generate the image
                image_path = generate_image(prompt, save_path)
                if image_path is None:
                    print(f"Failed to generate image for prompt: {prompt}")
                    continue

                # Perform face recognition to get the recognition score
                recognition_score = get_recognition_score(image_path, name, candidate_faces, candidate_names, tolerance=0.6)
                print(f"Recognition score for {name}: {recognition_score:.2f}")

                if recognition_score > 0.6:  # Retain images with a score > 60%
                    successful_images.append({"image_path": image_path, "prompt": prompt})
                else:
                    os.remove(image_path)  # Delete unrecognized images

            # Save the successful images and prompts to the results file
            if successful_images:
                with open(results_file, "a", encoding="utf-8") as results:
                    for item in successful_images:
                        json.dump(
                            {"name": name, "category": category, "prompt": item["prompt"], "image_path": item["image_path"]},
                            results,
                            ensure_ascii=False,
                        )
                        results.write("\n")

    print("Image generation process complete.")

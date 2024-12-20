import face_recognition
import os

def load_candidate_faces(dataset_path):

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
                                candidate_faces.append(encodings[0]) 
                                candidate_names.append(f"{category}/{celebrity}")  
                            break  # Select only one image per celebrity

    if not candidate_faces:
        raise ValueError("No valid faces found in the dataset. Please check the dataset structure and image quality.")

    return candidate_faces, candidate_names


def recognize_face(candidate_faces, candidate_names, test_image_path, tolerance=0.6):

    # Load the test image
    test_image = face_recognition.load_image_file(test_image_path)
    test_encodings = face_recognition.face_encodings(test_image)

    if not test_encodings:
        return False, "No face detected in the test image."

    test_encoding = test_encodings[0]

    # Compare the test image encoding with candidate face encodings
    results = face_recognition.compare_faces(candidate_faces, test_encoding, tolerance)
    if True in results:
        match_index = results.index(True)
        return True, candidate_names[match_index]  # Match found, return True and the name
    else:
        return False, "No Match"  # No match found


if __name__ == "__main__":
    # Path to the dataset
    dataset_path = "Celebrity_Image_Dataset"
    test_image_path = "test_image/test2.jpeg"  # Path to the new image to be tested

    # Load candidate images (one per celebrity)
    candidate_faces, candidate_names = load_candidate_faces(dataset_path)

    # Recognize face in the new image
    is_detected, result = recognize_face(candidate_faces, candidate_names, test_image_path)

    # Output the result
    if is_detected:
        print(f"Detected Celebrity: {result}")
    else:
        print(f"Result: {result}")
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from sklearn.preprocessing import LabelEncoder
from PIL import Image
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau, OneCycleLR  
import json
import google.generativeai as genai
import face_recognition
import numpy as np
import pickle

os.environ["gemini_API_KEY"] = "AIzaSyCgyxlDsqFlffNhj1ZL5P-d44E88mC6E2I"

# Add face extraction function
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


class CelebrityDataset(Dataset):
    def __init__(self, image_paths, labels, categories, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.categories = categories
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        label = self.labels[idx]
        category = self.categories[idx]

        if self.transform:
            image = self.transform(image)

        return image, label, category



def load_dataset(dataset_path):
    image_paths = []
    labels = []
    categories = []
    category_encoder = {}
    current_category_index = 0

    for category in os.listdir(dataset_path):
        category_path = os.path.join(dataset_path, category)
        if os.path.isdir(category_path):
            if category not in category_encoder:
                category_encoder[category] = current_category_index
                current_category_index += 1

            for label in os.listdir(category_path):
                label_path = os.path.join(category_path, label)
                if os.path.isdir(label_path):
                    for img_file in os.listdir(label_path):
                        if img_file.endswith(('jpg', 'jpeg', 'png')):
                            img_path = os.path.join(label_path, img_file)
                            # extrct face
                            face_image = extract_face(img_path)
                            if face_image is not None:
                                # Save the face image
                                face_path = os.path.join(label_path, f'face_{img_file}')
                                face_image.save(face_path)
                                image_paths.append(face_path)
                                labels.append(label)
                                categories.append(category_encoder[category])

    if len(image_paths) == 0:
        raise ValueError("Dataset is empty or path is incorrect. Please check the dataset structure.")
    return image_paths, labels, categories, category_encoder


# EfficientNet 
class EfficientNetClassifier(nn.Module):
    def __init__(self, num_classes):
        super(EfficientNetClassifier, self).__init__()
        # Load pre-trained EfficientNet-B0
        self.efficientnet = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        # Replace the last fully connected layer to adapt to num_classes
        self.efficientnet.classifier[1] = nn.Linear(self.efficientnet.classifier[1].in_features, num_classes)

    def forward(self, x):
        return self.efficientnet(x)


def train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=10, early_stop_patience=15):
    best_val_accuracy = 0.0
    patience_counter = 0
    best_model_state = None

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        with tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch") as tepoch:
            for images, labels, _ in tepoch:
                images = images.to(device)
                labels = labels.to(device).long()

                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                tepoch.set_postfix(loss=loss.item())

        avg_train_loss = running_loss / len(train_loader)
        val_accuracy = validate_model(model, val_loader, device)

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_train_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")

         # early stopping
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            
        # if the validation accuracy does not improve for a long time, stop training
        if patience_counter >= early_stop_patience:
            print(f"\nEarly stopping triggered after {epoch+1} epochs!")
            print(f"Best validation accuracy: {best_val_accuracy:.4f}")
            break

    # load the best model state
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        save_label_encoder(label_encoder)
        torch.save(best_model_state, "best_model.pth")
        print("Training finished. Best model saved as 'best_model.pth'.")
    else:
        save_label_encoder(label_encoder)
        torch.save(model.state_dict(), "best_model.pth")
        print("Training finished. Final model saved as 'best_model.pth'.")

# after training, save the label encoder
def save_label_encoder(label_encoder, filename='label_encoder.pkl'):
    with open(filename, 'wb') as f:
        pickle.dump(label_encoder, f)

# 加load_label_encoder函数
def load_label_encoder(filename='label_encoder.pkl'):
    with open(filename, 'rb') as f:
        return pickle.load(f)


def validate_model(model, val_loader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels, _ in val_loader:
            images = images.to(device)
            labels = labels.to(device).long()

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    return accuracy


# face_recog Function
def gemini_api_call(image_path):
    try:
        # Configure API key
        GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")  # Get API key from environment variable
        if not GEMINI_API_KEY:
            raise ValueError("Please set GEMINI_API_KEY environment variable")
            
        genai.configure(api_key=GEMINI_API_KEY)

        # Load Gemini Pro Vision model
        model = genai.GenerativeModel('gemini-pro-vision')

        # Open and prepare the image
        image = Image.open(image_path)
        
        # Construct prompt
        prompt = """Please analyze this image and tell me if there are any celebrities in it.
        If you see a celebrity, return their name.
        If not, indicate that no celebrities were found.
        Return in JSON format only: {"celebrity_detected": true/false, "name": "celebrity name or None"}"""

        # Call API
        response = model.generate_content([prompt, image])
        response_text = response.text
        
        # Process response
        if "celebrity_detected" in response_text:
            try:
                # Use json.loads() for safe JSON parsing
                return json.loads(response_text)
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON response: {e}")
                return {"celebrity_detected": False, "name": None}
        else:
            if "no" in response_text.lower() or "not found" in response_text.lower():
                return {"celebrity_detected": False, "name": None}
            else:
                return {"celebrity_detected": True, "name": response_text.strip()}

    except Exception as e:
        print(f"API call error: {e}")
        return {"celebrity_detected": False, "name": None}

def face_recog(image_path, model=None, label_encoder=None, device='cuda', confidence_threshold=0, version='local',):
    if version == 'gemini':
        # Use external API for detection
        response = gemini_api_call(image_path)
        if response["celebrity_detected"]:
            return response["name"], 1.0
        else:
            return "No Celebrity Detected"
    
    elif version == 'local':
        face_image = extract_face(image_path)
        model.eval()
        transform = transforms.Compose([
            transforms.Resize((160, 160)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

        image = transform(face_image).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(image)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)

        if confidence.item() < confidence_threshold:
            return "No Celebrity Detected (Low Confidence)", confidence.item()
        else:
            predicted_label = label_encoder.inverse_transform([predicted.item()])[0]
            return predicted_label, confidence.item()


def stratified_split_by_person(image_paths, labels, categories, train_ratio=0.8):
    """
    Hierarchical division by name to ensure that each person 's picture 
    is divided according to the specified proportion.
    """

    person_to_indices = {}
    for idx, label in enumerate(labels):
        if label not in person_to_indices:
            person_to_indices[label] = []
        person_to_indices[label].append(idx)
    
    train_indices = []
    val_indices = []
    
    for person, indices in person_to_indices.items():
        n_samples = len(indices)
        n_train = int(n_samples * train_ratio)
        

        indices = np.random.permutation(indices)
        

        train_indices.extend(indices[:n_train])
        val_indices.extend(indices[n_train:])
    

    train_dataset = CelebrityDataset(
        [image_paths[i] for i in train_indices],
        [labels[i] for i in train_indices],
        [categories[i] for i in train_indices],
        transform=transforms.Compose([
            transforms.RandomResizedCrop((160, 160), scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(30),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
    )
    
    val_dataset = CelebrityDataset(
        [image_paths[i] for i in val_indices],
        [labels[i] for i in val_indices],
        [categories[i] for i in val_indices],
        transform=transforms.Compose([
            transforms.Resize((160, 160)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
    )
    
    return train_dataset, val_dataset

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # dataset_path = 'Celebrity_Image_Dataset'
    # image_paths, labels, categories, category_encoder = load_dataset(dataset_path)

    # label_encoder = LabelEncoder()
    # encoded_labels = label_encoder.fit_transform(labels)

    # transform = transforms.Compose([
    #     transforms.RandomResizedCrop((160, 160), scale=(0.8, 1.0)),
    #     transforms.RandomHorizontalFlip(),
    #     transforms.RandomRotation(30),
    #     transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
    #     transforms.ToTensor(),
    #     transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    # ])

    # # partition function
    # train_dataset, val_dataset = stratified_split_by_person(image_paths, encoded_labels, categories)

    # train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    # val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)
    

    label_encoder = load_label_encoder()
    num_classes = len(label_encoder.classes_)
    model = EfficientNetClassifier(num_classes=num_classes).to(device)

    # criterion = nn.CrossEntropyLoss()
    # optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=100)

    model.load_state_dict(torch.load("best_model.pth"))
    image_path = 'test_image/test15.jpeg'
    celebrity_name, confidence = face_recog(image_path, model, label_encoder, device=device)
    print("Detected Celebrity:", celebrity_name, "with confidence:", confidence)
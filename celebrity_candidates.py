import os
import numpy as np
import torch
from facenet_pytorch import InceptionResnetV1
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])


class HierarchicalCelebrityDataset(Dataset):
    def __init__(self, dataset_path, transform=None):

        self.image_paths = []
        self.labels = []
        self.categories = []
        self.transform = transform

        for category in os.listdir(dataset_path):
            category_path = os.path.join(dataset_path, category)
            if os.path.isdir(category_path):
                for person in os.listdir(category_path):
                    person_path = os.path.join(category_path, person)
                    if os.path.isdir(person_path):
                        for img_name in os.listdir(person_path):
                            if img_name.endswith(('jpg', 'jpeg', 'png')):
                                self.image_paths.append(os.path.join(person_path, img_name))
                                self.labels.append(person)
                                self.categories.append(category)


        self.label_encoder = LabelEncoder()
        self.labels = self.label_encoder.fit_transform(self.labels)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        category = self.categories[idx]

        image = Image.open(image_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label, category

    def get_label_encoder(self):
        return self.label_encoder


class EfficientNetClassifier(nn.Module):
    def __init__(self, num_classes):
        super(EfficientNetClassifier, self).__init__()
        # Load pre-trained EfficientNet-B0
        self.efficientnet = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        # Replace the last fully connected layer to adapt to num_classes
        self.efficientnet.classifier[1] = nn.Linear(self.efficientnet.classifier[1].in_features, num_classes)

    def forward(self, x):
        return self.efficientnet(x)
    
# extract features
def extract_features(image_path, model, transform, device):
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        outputs = model(image)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
    return probabilities.cpu().numpy()

def build_candidate_features(dataset_path, model, transform, device):
    candidate_features = []
    candidate_labels = []
    for category in os.listdir(dataset_path):
        category_path = os.path.join(dataset_path, category)
        if os.path.isdir(category_path):
            for person in os.listdir(category_path):
                person_path = os.path.join(category_path, person)
                if os.path.isdir(person_path):
                    for image_file in os.listdir(person_path):
                        if image_file.endswith(('jpg', 'jpeg', 'png')):
                            image_path = os.path.join(person_path, image_file)
                            embedding = extract_features(image_path, model, transform, device)
                            candidate_features.append(embedding)
                            candidate_labels.append(f"{category}/{person}")
    return np.vstack(candidate_features), candidate_labels

def find_top_candidates(query_image_path, model, label_encoder, transform, device, top_n=5):
    probabilities = extract_features(query_image_path, model, transform, device)
    top_indices = np.argsort(-probabilities[0])[:top_n]  # 找到概率最高的前 top_n 类别
    top_candidates = [(label_encoder.inverse_transform([idx])[0], probabilities[0][idx]) for idx in top_indices]
    return top_candidates

if __name__ == "__main__":
    dataset_path = 'Celebrity_Image_Dataset'


    dataset = HierarchicalCelebrityDataset(dataset_path, transform=transform)
    
    label_encoder = dataset.get_label_encoder()

    num_classes = len(dataset.get_label_encoder().classes_)
    model = EfficientNetClassifier(num_classes=num_classes).to(device)
    

    model.load_state_dict(torch.load("best_model.pth"))


    print("Building candidate features...")
    candidate_features, candidate_labels = build_candidate_features(dataset_path, model, transform, device)
    print("Candidate features built.")


    query_image_path = 'test_image/test17.jpg'  

    print(f"Finding top candidates for: {query_image_path}")
    top_candidates = find_top_candidates(query_image_path, model, label_encoder, transform, device, top_n=5)

    print("Top Candidates:")
    for person, distance in top_candidates:
        print(f"Candidate: {person}, Distance: {distance:.4f}")

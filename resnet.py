import os
import numpy as np
import torch
import pandas as pd
from PIL import Image
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision import models

# Data Preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Dataset and DataLoader
test_dir = "/kaggle/input/vlg-recruitment-24-challenge/vlg-dataset/test"
train_dir = "/kaggle/input/vlg-recruitment-24-challenge/vlg-dataset/train"
train_dataset = ImageFolder(root=train_dir, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

print("Class to Index Mapping:", train_dataset.class_to_idx)

# Model Definition
class Model(nn.Module):
    def __init__(self, num_classes):
        super(Model, self).__init__()
        
        # Using ResNet18 as backbone with pretrained weights
        self.backbone = models.resnet18(pretrained=True)
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(self.backbone.classifier[1].in_features, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)

# Training Setup
num_classes = len(train_dataset.classes)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = Model(num_classes).to(device)

# initialize loss function and optimizer
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=3e-4,
    weight_decay=1e-4
)

# Training Loop
num_epochs = 15
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    # iterate over the training data
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        # clear gradients, forward pass, calculate loss, backpropagation, update weights
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        correct += (torch.argmax(outputs, dim=1) == labels).sum().item()
        total += labels.size(0)

    # Calculate training accuracy and loss
    train_accuracy = correct / total
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss:.4f}, Accuracy: {train_accuracy:.4f}")



# Prediction
model.eval()
test_images = [f for f in os.listdir(test_dir) if f.endswith('.jpg')]
test_predictions = []

for img_name in test_images:
    img_path = os.path.join(test_dir, img_name)
    image = Image.open(img_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image)
        predicted_class = torch.argmax(outputs, dim=1).item()
        test_predictions.append((img_name, train_dataset.classes[predicted_class]))

# Save Predictions
submission = pd.DataFrame(test_predictions, columns=['image_id', 'class'])
submission.to_csv("/kaggle/working/predictions.csv", index=False)


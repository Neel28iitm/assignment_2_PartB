# Step 1: Install Libraries
!pip install -q torchvision wandb

import wandb
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, random_split
import os
from google.colab import drive
import zipfile

# Step 2: Mount Drive and Extract Dataset
drive.mount('/content/drive')
zip_path = "/content/drive/MyDrive/Zolvit_Neel/nature_12K.zip"
extract_path = "/content/inaturalist_data"

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_path)

# Step 3: Define Transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Step 4: Load Dataset
dataset = datasets.ImageFolder(root=extract_path, transform=transform)
val_size = int(0.2 * len(dataset))
train_size = len(dataset) - val_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)

# Step 5: Load Pretrained Model (ResNet50)
model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

# Freeze all layers
for param in model.parameters():
    param.requires_grad = False

# Replace classifier
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 10)

# Unfreeze final layers for fine-tuning
for param in model.layer4.parameters():
    param.requires_grad = True
for param in model.fc.parameters():
    param.requires_grad = True

# Step 6: Set Loss, Optimizer, Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)

# Step 7: Train Function
def train_model(model, criterion, optimizer, train_loader, val_loader, epochs=10):
    wandb.init(project="inat-finetune-resnet50", name="resnet50-run-final")
    wandb.watch(model, log="all")

    for epoch in range(epochs):
        model.train()
        train_loss, train_correct = 0.0, 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)
            train_correct += (outputs.argmax(1) == labels).sum().item()

        model.eval()
        val_loss, val_correct = 0.0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                val_correct += (outputs.argmax(1) == labels).sum().item()

        print(f"Epoch {epoch+1}/{epochs} | Train Acc: {train_correct/train_size:.4f} | Val Acc: {val_correct/val_size:.4f}")

    torch.save(model.state_dict(), "resnet50_finetuned.pth")
    wandb.finish()

train_model(model, criterion, optimizer, train_loader, val_loader, epochs=10)

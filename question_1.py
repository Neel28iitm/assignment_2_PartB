# Step 1: Install libraries
!pip install -q torchvision wandb

import wandb
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, random_split
import os

# Step 2: Define Transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Step 3: Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

zip_path = "/content/drive/MyDrive/Zolvit_Neel/nature_12K.zip"
extract_path = "/content/inaturalist_data"

import zipfile
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_path)

# Step 4: Loading Dataset
dataset = datasets.ImageFolder(root=extract_path, transform=transform)

# Split into train and test 
val_size = int(0.2 * len(dataset))
train_size = len(dataset) - val_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)

# Step 5: Load Pre-trained ResNet50 and Modify Final Layer
model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

# Freeze all layers
for param in model.parameters():
    param.requires_grad = False

# Replace the final fully connected layer
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 10) 

# Unfreeze final layers for fine-tuning
for param in model.layer4.parameters():
    param.requires_grad = True
for param in model.fc.parameters():
    param.requires_grad = True

# Step 6: Set Up Loss, Optimizer, and Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)

# Step 7: Training Function
def train_model(model, criterion, optimizer, train_loader, val_loader, epochs=10):
    wandb.init(project="inat-finetune-resnet50", name="resnet50-run-final")
    wandb.watch(model, log="all", log_freq=100)

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
            _, preds = torch.max(outputs, 1)
            train_correct += torch.sum(preds == labels)

        model.eval()
        val_loss, val_correct = 0.0, 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                val_correct += torch.sum(preds == labels)

        epoch_train_acc = train_correct.double() / train_size
        epoch_val_acc = val_correct.double() / val_size

        # Log metrics to wandb
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss / train_size,
            "val_loss": val_loss / val_size,
            "train_acc": epoch_train_acc.item(),
            "val_acc": epoch_val_acc.item()
        })

        print(f"Epoch {epoch+1}/{epochs} | Train Acc: {epoch_train_acc:.4f} | Val Acc: {epoch_val_acc:.4f}")

    # Save model to wandb
    torch.save(model.state_dict(), "resnet50_finetuned.pth")
    wandb.save("resnet50_finetuned.pth")
    wandb.finish()

# Step 8: Train the Model
train_model(model, criterion, optimizer, train_loader, val_loader, epochs=10)

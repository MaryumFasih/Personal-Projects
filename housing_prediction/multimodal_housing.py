# multimodal_housing.py

import os
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

# -------------------------------
# 1. Config
# -------------------------------
IMG_DIR = 'images'   # Folder with images named 1.jpg to 17.jpg
CSV_FILE = 'data.csv'
NUM_IMAGES = 17      # Number of images you have

BATCH_SIZE = 4
EPOCHS = 5
LR = 0.001
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# -------------------------------
# 2. Load & fix data
# -------------------------------
df = pd.read_csv(CSV_FILE)

# Trim to match images
df = df.iloc[:NUM_IMAGES].copy().reset_index(drop=True)

# Ensure image_id matches your files: 1.jpg to 17.jpg
df['image_id'] = range(1, NUM_IMAGES + 1)

print(df[['image_id', 'Lot Area', 'Overall Qual', 'SalePrice']].head())

tabular_features = ['Lot Area', 'Overall Qual']  # Use any numeric columns you like
target = 'SalePrice'

image_ids = df['image_id'].astype(int).values
X_tab = df[tabular_features].values
y = df[target].values

# Scale tabular features
scaler = StandardScaler()
X_tab = scaler.fit_transform(X_tab)

# Train/test split
X_tab_train, X_tab_test, y_train, y_test, img_train, img_test = train_test_split(
    X_tab, y, image_ids, test_size=0.2, random_state=42
)

# -------------------------------
# 3. Dataset
# -------------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

class HousingDataset(Dataset):
    def __init__(self, image_ids, tabular_data, targets, img_dir, transform=None):
        self.image_ids = image_ids
        self.tabular_data = torch.tensor(tabular_data, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        img_path = os.path.join(self.img_dir, f"{img_id}.jpg")
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image file not found: {img_path}")
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        tabular = self.tabular_data[idx]
        target = self.targets[idx]
        return image, tabular, target

train_dataset = HousingDataset(img_train, X_tab_train, y_train, IMG_DIR, transform)
test_dataset = HousingDataset(img_test, X_tab_test, y_test, IMG_DIR, transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

# -------------------------------
# 4. Model
# -------------------------------
class MultimodalNet(nn.Module):
    def __init__(self, tabular_input_dim):
        super(MultimodalNet, self).__init__()
        self.cnn = models.resnet18(weights="IMAGENET1K_V1")
        self.cnn.fc = nn.Identity()  # Remove last FC

        self.tabular_mlp = nn.Sequential(
            nn.Linear(tabular_input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32)
        )

        self.combined = nn.Sequential(
            nn.Linear(512 + 32, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, image, tabular):
        img_features = self.cnn(image)
        tabular_features = self.tabular_mlp(tabular)
        combined = torch.cat((img_features, tabular_features), dim=1)
        output = self.combined(combined)
        return output.squeeze()

model = MultimodalNet(tabular_input_dim=X_tab.shape[1]).to(DEVICE)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# -------------------------------
# 5. Train
# -------------------------------
print("\n✅ Training started...")
for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0.0
    for images, tabular, targets in train_loader:
        images, tabular, targets = images.to(DEVICE), tabular.to(DEVICE), targets.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(images, tabular)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * images.size(0)
    epoch_loss /= len(train_dataset)
    print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {epoch_loss:.2f}")

# -------------------------------
# 6. Evaluate + plot
# -------------------------------
print("\n✅ Evaluating on test set...")
model.eval()
preds = []
targets_list = []
with torch.no_grad():
    for images, tabular, targets in test_loader:
        images, tabular = images.to(DEVICE), tabular.to(DEVICE)
        outputs = model(images, tabular)
        preds.extend(outputs.cpu().numpy())
        targets_list.extend(targets.numpy())

mae = mean_absolute_error(targets_list, preds)
rmse = np.sqrt(mean_squared_error(targets_list, preds))
print(f"\n✅ MAE: {mae:.2f}")
print(f"✅ RMSE: {rmse:.2f}")

# -------------------------------
# 7. Save model
# -------------------------------
torch.save(model.state_dict(), "multimodal_model.pth")
print("\n✅ Model saved as 'multimodal_model.pth'")

# -------------------------------
# 8. Plot true vs predicted
# -------------------------------
plt.figure(figsize=(6, 6))
plt.scatter(targets_list, preds, alpha=0.7)
plt.plot([min(targets_list), max(targets_list)],
         [min(targets_list), max(targets_list)],
         color='red', linestyle='--')
plt.xlabel("True Price")
plt.ylabel("Predicted Price")
plt.title("True vs Predicted House Prices")
plt.grid(True)
plt.show()


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import itertools
import os

#Config / device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# reproducibility
torch.manual_seed(42)
np.random.seed(42)
if device.type == "cuda":
    torch.cuda.manual_seed_all(42)

#Transforms & Datasets 
transform = transforms.Compose([
    transforms.ToTensor(),           
])

train_full = datasets.FashionMNIST(root='data', train=True, download=True, transform=transform)
test_set = datasets.FashionMNIST(root='data', train=False, download=True, transform=transform)

# split train/val
val_len = int(0.1 * len(train_full))
train_len = len(train_full) - val_len
train_set, val_set = random_split(train_full, [train_len, val_len])

# DataLoader args 
batch_size = 128

num_workers = 2
pin_memory = True if device.type == "cuda" else False




train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                          num_workers=num_workers, pin_memory=pin_memory)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=pin_memory)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False,
                         num_workers=num_workers, pin_memory=pin_memory)

# Model
class FashionCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(2)
        self.drop = nn.Dropout(0.25)

        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.drop2 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))
        x = self.drop(x)
        x = self.pool(torch.relu(self.bn2(self.conv2(x))))
        x = self.drop(x)
        x = x.view(x.size(0), -1)
        x = torch.relu(self.bn3(self.fc1(x)))
        x = self.drop2(x)
        x = self.fc2(x)
        return x

model = FashionCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Training / Evaluation functions
def train_epoch(loader):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for xb, yb in loader:
        xb, yb = xb.to(device, non_blocking=pin_memory), yb.to(device, non_blocking=pin_memory)
        optimizer.zero_grad()
        out = model(xb)
        loss = criterion(out, yb)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * xb.size(0)
        _, preds = out.max(1)
        correct += (preds == yb).sum().item()
        total += xb.size(0)
    return running_loss / total, correct / total

def eval_loader(loader):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

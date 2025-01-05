import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model.VGG import *
from utils.utils import *
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
#from sklearn.metrics import confusion_matrix

# Prepare CIFAR-10 data
transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

train_dataset = datasets.CIFAR10(root='../data', train=True, download=True, transform=transform)
test_dataset = datasets.CIFAR10(root='../data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # Use GPU if available
print(f"Using device: {device}")

model = VGGNet().to(device)
model_path = '../results/VGG_cifar10.pth'
loss_path='../results/VGG_loss_curve.png'
# 检查模型文件是否存在
if os.path.exists(model_path):
    # 如果存在，加载模型
    print("Model found, loading...")
    model.load_state_dict(torch.load(model_path))
else:
    # 如果不存在，进行训练
    print("Model not found, starting training...")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_loss=[]
    print("Starting training...")
    for epoch in range(50):
        model.train()
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 100 == 99:
                print(f"Epoch {epoch + 1}, Batch {i + 1}, Loss: {running_loss / 100:.4f}")
                running_loss = 0.0

        avg_loss=running_loss/len(train_loader)
        train_loss.append(avg_loss)

        # 可视化损失曲线
        if (epoch + 1) % 10 == 0:
            plot_loss_curve(train_loss, loss_path)

    # 所有epoch完成后保存模型
    torch.save(model.state_dict(), model_path)
    print("Training completed and model saved.")

# Evaluation
print("Evaluating model...")
model.eval()
#correct = 0
#total = 0
all_labels = []
all_predictions = []

num_classes = 10
conf_matrix = torch.zeros(num_classes, num_classes)

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)  # Move data to GPU
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        #otal += labels.size(0)
        #correct += (predicted == labels).sum().item()
        all_labels.extend(labels.view(-1).cpu().numpy())
        all_predictions.extend(predicted.view(-1).cpu().numpy())


correct = conf_matrix.diag().sum().item()
total = conf_matrix.sum().item()
print(f"Test accuracy: {100 * correct / total:.2f}%")


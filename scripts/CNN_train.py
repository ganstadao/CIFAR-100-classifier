import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model.CNN import *
from utils.utils import *
import os

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

train_dataset = datasets.CIFAR10(root='../data', train=True, download=True, transform=transform_train)
test_dataset = datasets.CIFAR10(root='../data', train=False, download=True, transform=transform_test)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

model_path = '../results/CNN_cifar10.pth'
loss_path = '../results/CNN_loss_curve.png'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

model = CNN(num_classes=10).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.01)

scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

if os.path.exists(model_path):
    # 如果存在，加载模型
    print("Model found, loading...")
    model.load_state_dict(torch.load(model_path))
else:
    print("Starting training...")
    train_loss = []
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

        avg_loss = running_loss / len(train_loader)
        train_loss.append(avg_loss)

        # 可视化损失曲线
        if (epoch + 1) % 10 == 0:
            plot_loss_curve(train_loss, loss_path)

        scheduler.step()

    torch.save(model.state_dict(), model_path)
    print("Model saved as cnn_cifar10.pth")

print("Evaluating model...")
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Test accuracy: {100 * correct / total:.2f}%")

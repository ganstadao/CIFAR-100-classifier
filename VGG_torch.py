import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
#from sklearn.metrics import confusion_matrix

# Define the VGG-like model
class VGGNet(nn.Module):
    def __init__(self):
        super(VGGNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512), nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 512), nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

def visualize_prediction(model, dataloader, device, index=None):
    # Set the model to evaluation mode
    model.eval()

    # Get a batch of data
    dataiter = iter(dataloader)
    images, labels = dataiter.next()

    # Move data to the appropriate device
    images, labels = images.to(device), labels.to(device)

    # Get predictions
    with torch.no_grad():
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

    # Choose a random image if index is not specified
    if index is None:
        index = np.random.randint(0, len(images))

    # Display the image and the labels
    plt.imshow(images[index].cpu().numpy().transpose((1, 2, 0)))
    plt.title(f'Predicted: {predicted[index].item()}, Actual: {labels[index].item()}')
    plt.axis('off')  # Hide axes
    plt.show()

# Prepare CIFAR-10 data
transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Initialize model, loss function, and optimizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # Use GPU if available
print(f"Using device: {device}")

model = VGGNet().to(device)
model_path = '../VGG_cifar10.pth'
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

    # Training loop
    print("Starting training...")
    for epoch in range(10):
        model.train()
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)  # Move data to GPU

            # Zero the gradient buffers
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 100 == 99:  # Print every 100 mini-batches
                print(f"Epoch {epoch + 1}, Batch {i + 1}, Loss: {running_loss / 100:.4f}")
                running_loss = 0.0

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

# 初始化混淆矩阵
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

        # 更新混淆矩阵
        for t, p in zip(labels.view(-1), predicted.view(-1)):
            conf_matrix[t.long(), p.long()] += 1

correct = conf_matrix.diag().sum().item()
total = conf_matrix.sum().item()
print(f"Test accuracy: {100 * correct / total:.2f}%")

# 将混淆矩阵转换为numpy数组
conf_matrix = conf_matrix.cpu().numpy()

# 绘制混淆矩阵的热力图
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Confusion Matrix Heatmap')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.xticks(np.arange(num_classes))
plt.yticks(np.arange(num_classes))
plt.savefig('Confusion Matrix Heatmap.png')
plt.show()

visualize_prediction(model,test_loader,device)

import torch
import os
from model.resnet import ResNet, ResidualBlock
from model.VGG import *
from model.CNN import *
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from utils.utils import plot_confusion_matrix

# 准备数据
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

test_dataset = datasets.CIFAR10(root='../data', train=False, download=True, transform=transform_test)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

model_select = "VGG"
model_path = f"../results/{model_select}_cifar10.pth"
confusion_path = f"../results/{model_select}_confusion_matrix.png"

# 加载模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = VGGNet().to(device)
if model_select == "VGG":
    model = VGGNet().to(device)
elif model_select == "resnet":
    model = ResNet(ResidualBlock, [2, 2, 2, 2]).to(device)
elif model_select == "CNN":
    model = CNN(num_classes=10).to(device)

if os.path.exists(model_path):
    print("model found, loading...")
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()
    print("success loaded, evaluating...")
    # 评估模型
    true_labels = []
    pred_labels = []
    class_names = test_dataset.classes

    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)

            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            true_labels.extend(labels.cpu().numpy())
            pred_labels.extend(predicted.cpu().numpy())

    # 可视化混淆矩阵
    plot_confusion_matrix(true_labels, pred_labels, class_names, confusion_path)
    print(f"Test Accuracy: {100 * correct / total}%")

else:
    print("model not found, please train!")

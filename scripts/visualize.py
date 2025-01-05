import torch
import os
from model.resnet import ResNet, ResidualBlock
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from utils.utils import visualize_random_images

# 准备数据
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

test_dataset = datasets.CIFAR10(root='../data', train=False, download=True, transform=transform_test)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

model_path="../results/resnet_cifar10.pth"

# 加载模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ResNet(ResidualBlock, [2, 2, 2, 2]).to(device)
if os.path.exists(model_path):
    print("model found, loading...")
    model.load_state_dict(torch.load("../results/resnet_cifar10.pth",weights_only=True))
    print("success loaded, visualizing...")
    times = int(input("input mounts you want to test:"))
    for _ in range(times):
        class_names = test_dataset.classes
        visualize_random_images(model, test_loader, device, class_names)

else:
    print("model not found, please train!")


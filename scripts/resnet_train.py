import torch
import torch.optim as optim
import torch.nn as nn
from model.resnet import ResNet, ResidualBlock
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
from utils.utils import plot_loss_curve

# 准备数据
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

train_dataset = datasets.CIFAR10(root='../data', train=True, download=True, transform=transform_train)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

# 设置设备和模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ResNet(ResidualBlock, [2, 2, 2, 2]).to(device)
model_path="../results/resnet_cifar10.pth"
loss_path= "../results/resnet_loss_curve.png"
print(f"using {device}")

# 损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)

if os.path.exists(model_path):
    print("model found, loading...")
    model.load_state_dict(torch.load(model_path, weights_only=True))
    print("loading success!")
else:
    print("model not found!")

    # 训练过程
    print("start training...")
    train_loss = []
    for epoch in range(100):
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

        avg_loss = running_loss / len(train_loader)
        train_loss.append(avg_loss)
        print(f"Epoch [{epoch + 1}/100], Loss: {avg_loss:.4f}")

        # 可视化损失曲线
        if (epoch + 1) % 10 == 0:
            plot_loss_curve(train_loss,loss_path)

    # 保存模型
    print("finish training, saving...")
    torch.save(model.state_dict(), "../results/resnet_cifar10.pth")

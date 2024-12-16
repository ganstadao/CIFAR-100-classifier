import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
from sklearn.metrics import confusion_matrix


# 训练损失图
def plot_loss_curve(train_loss, filename="../results/loss_curve.png"):
    plt.plot(train_loss)
    plt.title("Training Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.savefig(filename)
    plt.show()


# 混淆矩阵可视化
def plot_confusion_matrix(y_true, y_pred, class_names, filename="../results/confusion_matrix.png"):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.savefig(filename)
    plt.show()


# 随机显示图片和标签
def visualize_random_images(model, test_loader, device, class_names):
    model.eval()
    data_iter = iter(test_loader)
    images, labels = next(data_iter)
    images, labels = images.to(device), labels.to(device)

    # Randomly select an image
    idx = np.random.randint(0, len(images))
    image = images[idx].cpu().numpy().transpose((1, 2, 0))
    true_label = labels[idx].item()

    model_output = model(images)
    _, predicted = torch.max(model_output, 1)
    predicted_label = predicted[idx].item()

    # Plot image
    plt.imshow(image)
    plt.title(f"True: {class_names[true_label]}, Pred: {class_names[predicted_label]}")
    plt.axis('off')
    plt.show()

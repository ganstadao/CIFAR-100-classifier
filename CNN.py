import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pickle
import os

# 加载batches数据（官网提供）
def load_cifar10_batch(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    data = dict[b'data']
    labels = dict[b'labels']
    return data, labels

# 数据加载
def load_cifar10(data_dir):
    train_data, train_labels = [], []
    #加载1-5的数据
    for i in range(1, 6):
        file = os.path.join(data_dir, f'data_batch_{i}')
        data, labels = load_cifar10_batch(file)
        train_data.append(data)
        train_labels.extend(labels)

    train_data = np.vstack(train_data)
    train_labels = np.array(train_labels)

    # Load test data
    test_file = os.path.join(data_dir, 'test_batch')
    test_data, test_labels = load_cifar10_batch(test_file)

    return train_data, train_labels, np.array(test_data), np.array(test_labels)

#呈现数据
def show_img(data,labels):
    for i in range(9):
        # 显示图像
        plt.imshow(data[i])
        plt.title(f"Label: {labels[i]}")
        plt.axis('off')  # 不显示坐标轴
        plt.show()

# 预处理数据（转为可以处理的数据格式+数据增强）
def preprocess_data(data):
    data = data.reshape(-1, 3, 32, 32)  # 分解为为(10000, 3, 32, 32)（numpy图像数组格式）
    data = data.transpose(0, 2, 3, 1)   # 转为(10000, 32, 32, 3)

    data = data.astype('float32') / 255.0  #归一化

    #数据增强使用的是ImageDataGenerator

    return data

# 搭建CNN模型
def create_cnn_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    return model

if __name__ == "__main__":
    '''# 检查TensorFlow是否看到GPU
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

    # 简单的计算来测试GPU
    tf.debugging.set_log_device_placement(True)

    # 创建一些张量
    a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

    # 使用`MatMul`运算符创建一个矩阵乘法运算，并将其放在GPU上执行
    c = tf.matmul(a, b)

    print(c)'''
    data_dir = "E:\AI\data\cifar-10-python\cifar-10-batches-py"

    # 加载与预处理数据
    train_data, train_labels, test_data, test_labels = load_cifar10(data_dir)
    train_data = preprocess_data(train_data)
    test_data = preprocess_data(test_data)

    # 将labels转化为one-hot编码
    train_labels = tf.keras.utils.to_categorical(train_labels, 10)
    test_labels = tf.keras.utils.to_categorical(test_labels, 10)

    #show_img(train_data,train_labels)

    #建立模型
    model = create_cnn_model()

    # 编译模型
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # 训练模型
    model.fit(train_data, train_labels, epochs=10, batch_size=64, validation_split=0.2)

    # 评估模型
    test_loss, test_acc = model.evaluate(test_data, test_labels, verbose=2)
    print(f"Test accuracy: {test_acc}")





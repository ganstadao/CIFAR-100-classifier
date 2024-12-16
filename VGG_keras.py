import tensorflow as tf
import numpy as np
import pickle
import os

# Load data from CIFAR-10 batches
def load_cifar10_batch(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    data = dict[b'data']
    labels = dict[b'labels']
    print(f"Loaded {file}: {len(labels)} samples")  # Debug statement
    return data, labels

# Load all CIFAR-10 data
def load_cifar10(data_dir):
    train_data, train_labels = [], []
    for i in range(1, 6):  # Load batches 1 to 5
        file = os.path.join(data_dir, f'data_batch_{i}')
        data, labels = load_cifar10_batch(file)
        train_data.append(data)
        train_labels.extend(labels)
        print(f"Batch {i} loaded: {data.shape} data points")  # Debug statement

    train_data = np.vstack(train_data)
    train_labels = np.array(train_labels)
    print(f"Total training data shape: {train_data.shape}")  # Debug statement

    # Load test data
    test_file = os.path.join(data_dir, 'test_batch')
    test_data, test_labels = load_cifar10_batch(test_file)
    print(f"Test data loaded: {test_data.shape} data points")  # Debug statement

    return train_data, train_labels, np.array(test_data), np.array(test_labels)

# Preprocess data
def preprocess_data(data):
    print(f"Preprocessing data with shape: {data.shape}")  # Debug statement
    data = data.reshape(-1, 3, 32, 32)  # Reshape to (num_samples, 3, 32, 32)
    data = data.transpose(0, 2, 3, 1)   # Reorder to (num_samples, 32, 32, 3)
    data = data.astype('float32') / 255.0  # Normalize to [0, 1]
    print(f"Data reshaped to: {data.shape}")  # Debug statement
    return data

# Define VGG-like CNN model
def create_vgg_model():
    print("Creating VGG-like model...")  # Debug statement
    model = tf.keras.models.Sequential()

    # Convolutional layers
    model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 3)))
    model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))

    model.add(tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))

    model.add(tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))

    # Flatten and Dense layers
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(512, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(512, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(10, activation='softmax'))

    print("VGG-like model created successfully.")  # Debug statement
    return model

# Main script
if __name__ == "__main__":
    data_dir = "E:\AI\data\cifar-10-python\cifar-10-batches-py" # Path to CIFAR-10 data folder

    # Load and preprocess data
    print("Loading and preprocessing data...")  # Debug statement
    train_data, train_labels, test_data, test_labels = load_cifar10(data_dir)
    train_data = preprocess_data(train_data)
    test_data = preprocess_data(test_data)

    # Convert labels to one-hot encoding
    print("Converting labels to one-hot encoding...")  # Debug statement
    train_labels = tf.keras.utils.to_categorical(train_labels, 10)
    test_labels = tf.keras.utils.to_categorical(test_labels, 10)

    # Create model
    model = create_vgg_model()

    # Compile model
    print("Compiling model...")  # Debug statement
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # Train model
    print("Starting training...")  # Debug statement
    model.fit(train_data, train_labels, epochs=10, batch_size=64, validation_split=0.2)

    # Evaluate model
    print("Evaluating model...")  # Debug statement
    test_loss, test_acc = model.evaluate(test_data, test_labels, verbose=2)
    print(f"Test accuracy: {test_acc}")

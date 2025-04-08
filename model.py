import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

def load_dataset(dataset_path="dataset"):
    images, labels = [], []
    class_names = sorted(os.listdir(dataset_path))
    
    for class_id, class_name in enumerate(class_names):
        class_dir = os.path.join(dataset_path, class_name)
        for img_name in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img_name)
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.resize(img, (64, 64))
                images.append(img)
                labels.append(class_id)
    
    return np.array(images), np.array(labels), class_names

def create_model(input_shape=(64,64,3), num_classes=6):
    model = Sequential([
        Conv2D(32, (3,3), activation='relu', input_shape=input_shape),
        MaxPooling2D(2,2),
        Conv2D(64, (3,3), activation='relu'),
        MaxPooling2D(2,2),
        Conv2D(128, (3,3), activation='relu'),
        MaxPooling2D(2,2),
        Flatten(),
        Dense(256, activation='relu', kernel_regularizer='l2'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def train():
    X, y, classes = load_dataset()
    X = X.astype('float32') / 255.0
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    datagen = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True
    )
    
    model = create_model()
    model.fit(datagen.flow(X_train, y_train, batch_size=32),
              epochs=20,
              validation_data=(X_test, y_test),
              callbacks=[EarlyStopping(patience=3)])
    
    model.save('gesture_model.h5')
    np.save('class_names.npy', classes)

if __name__ == "__main__":
    train()

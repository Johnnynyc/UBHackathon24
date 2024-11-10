import numpy as np
import cv2
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical

# Path to your .npy files directory
data_dir = r"/data/"
categories = ["apple","alarm clock","airplane","axe","baseball bat"]  

def load_data():
    images = []
    labels = []
    for idx, category in enumerate(categories):
     
        category_path = os.path.join(data_dir, f"{category}.npy")
        drawings = np.load(category_path, allow_pickle=True)  

        for drawing in drawings:
           
            img = np.zeros((255, 255), np.uint8)
            if isinstance(drawing, list):  
                for stroke in drawing:
                 
                    if isinstance(stroke, list) and len(stroke) == 2:
                        x_coords, y_coords = stroke
                        
                        for i in range(len(x_coords) - 1):
                            start_point = (x_coords[i], y_coords[i])
                            end_point = (x_coords[i + 1], y_coords[i + 1])
                            cv2.line(img, start_point, end_point, 255, 2)

      
            img = cv2.resize(img, (28, 28))
            images.append(img)
            labels.append(idx)

 
    images = np.array(images).reshape(-1, 28, 28, 1) / 255.0
    labels = np.array(labels)
    return images, labels


X, y = load_data()



y = to_categorical(y, num_classes=len(categories))

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(len(categories), activation='softmax')  
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

#this will train the model
model.fit(X, y, epochs=10, validation_split=0.2, batch_size=64)


model.save("quickdraw-apple.h5")
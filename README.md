# Model Creation used in the Project

### Importing Libraries and Modules
```python
from keras.utils import to_categorical
from keras_preprocessing.image import load_img
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D
import os
import pandas as pd
import numpy as np
```
- **`to_categorical`**: Converts class vectors to binary class matrices.
- **`load_img`**: Loads an image from a file.
- **`Sequential`**: Sequential model in Keras.
- **Various Layers**: Used to create the neural network layers.
- **`os`**: Provides a way of using operating system-dependent functionality.
- **`pandas as pd`**: Data manipulation and analysis library.
- **`numpy as np`**: Library for numerical computations.

### Setting Directory Paths
```python
TRAIN_DIR = '/content/images/train'
TEST_DIR = '/content/images/validation'
```
- **`TRAIN_DIR`**: Path to the training images directory.
- **`TEST_DIR`**: Path to the testing/validation images directory.

### Creating DataFrames from Directory
```python
def createdataframe(dir):
    image_paths = []
    labels = []
    for label in os.listdir(dir):
        for imagename in os.listdir(os.path.join(dir, label)):
            image_paths.append(os.path.join(dir, label, imagename))
            labels.append(label)
        print(label, "completed")
    return image_paths, labels

train = pd.DataFrame()
train['image'], train['label'] = createdataframe(TRAIN_DIR)

test = pd.DataFrame()
test['image'], test['label'] = createdataframe(TEST_DIR)
```
- **`createdataframe`**: 
  - **`image_paths`**: List of image paths.
  - **`labels`**: List of corresponding labels.
  - Loops through each label folder in the directory, collects image paths, and labels.
- **`train`**: DataFrame to hold training images and labels.
- **`test`**: DataFrame to hold testing images and labels.

### Extracting Features from Images
```python
from tqdm.notebook import tqdm

def extract_features(images):
    features = []
    for image in tqdm(images):
        img = load_img(image, grayscale=True)
        img = np.array(img)
        features.append(img)
    features = np.array(features)
    features = features.reshape(len(features), 48, 48, 1)
    return features

train_features = extract_features(train['image'])
test_features = extract_features(test['image'])

x_train = train_features / 255.0
x_test = test_features / 255.0
```
- **`tqdm`**: Progress bar for loops.
- **`extract_features`**:
  - Converts images to grayscale.
  - Converts image to numpy array.
  - Reshapes the array to add a channel dimension.
  - Normalizes image data to the range [0, 1].
- **`x_train`**: Normalized training images.
- **`x_test`**: Normalized testing images.

### Encoding Labels
```python
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
le.fit(train['label'])

y_train = le.transform(train['label'])
y_test = le.transform(test['label'])

y_train = to_categorical(y_train, num_classes=7)
y_test = to_categorical(y_test, num_classes=7)
```
- **`LabelEncoder`**: Encodes labels as numeric values.
- **`le.fit`**: Fits the encoder to the training labels.
- **`le.transform`**: Transforms labels to encoded form.
- **`to_categorical`**: Converts encoded labels to binary class matrices.

### Building the Model
```python
model = Sequential()
# convolutional layers
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu', input_shape=(48, 48, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.4))

model.add(Conv2D(256, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.4))

model.add(Conv2D(512, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.4))

model.add(Conv2D(512, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.4))

model.add(Flatten())
# fully connected layers
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.3))
# output layer
model.add(Dense(7, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
```
- **Convolutional Layers**: Extract features from images.
- **MaxPooling**: Downsamples the image.
- **Dropout**: Prevents overfitting by randomly setting input units to 0.
- **Flatten**: Flattens the input.
- **Dense Layers**: Fully connected layers.
- **Output Layer**: Classifies into 7 classes using softmax.
- **`model.compile`**: Compiles the model with Adam optimizer and categorical crossentropy loss.

### Training the Model
```python
model.fit(x=x_train, y=y_train, batch_size=128, epochs=100, validation_data=(x_test, y_test))
```
- **`model.fit`**: Trains the model using training data and evaluates it using validation data.

### Saving the Model
```python
model_json = model.to_json()
with open("emotiondetector.json", 'w') as json_file:
    json_file.write(model_json)
model.save("emotiondetector.h5")
```
- **`model.to_json`**: Converts the model to JSON format.
- **`json_file.write`**: Saves the JSON model to a file.
- **`model.save`**: Saves the model weights to an H5 file.

### Loading the Model
```python
from keras.models import model_from_json

json_file = open("emotiondetector.json", "r")
model_json = json_file.read()
json_file.close()
model = model_from_json(model_json)
model.load_weights("emotiondetector.h5")
```
- **`model_from_json`**: Loads model architecture from JSON file.
- **`model.load_weights`**: Loads the model weights from an H5 file.

### Defining the Emotion Function
```python
label = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

def ef(image):
    img = load_img(image, grayscale=True)
    feature = np.array(img)
    feature = feature.reshape(1, 48, 48, 1)
    return feature / 255.0
```
- **`label`**: List of emotion labels.
- **`ef`**: Preprocesses an input image to match the model's input shape and normalizes it.

### Testing the Model with New Images
```python
import matplotlib.pyplot as plt

image = '/content/images/train/sad/42.jpg'
print("original image is of sad")
img = ef(image)
pred = model.predict(img)
pred_label = label[pred.argmax()]
print("model prediction is", pred_label)
plt.imshow(img.reshape(48, 48), cmap='gray')

image = '/content/images/train/disgust/299.jpg'
print("original image is of disgust")
img = ef(image)
pred = model.predict(img)
pred_label = label[pred.argmax()]
print("model prediction is", pred_label)
plt.imshow(img.reshape(48, 48), cmap='gray')

image = '/content/images/train/happy/7.jpg'
print("original image is of happy")
img = ef(image)
pred = model.predict(img)
pred_label = label[pred.argmax()]
print("model prediction is", pred_label)
plt.imshow(img.reshape(48, 48), cmap='gray')
```
- **`plt.imshow`**: Displays the image.
- Tests the model's predictions on new images and compares them to the original labels.

In summary, the code builds, trains, and evaluates a convolutional neural network for emotion detection from facial images, saving the trained model for future use. It includes preprocessing steps, model architecture, training process, and prediction on new images.

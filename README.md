## Model 1, For Facial Expressions
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

## Model 2, For EEG Signals
### Importing Libraries and Modules
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report
```

### 1. **Load Data and Plot a Sample**
```python
data = pd.read_csv('../content/emotions.csv')
sample = data.loc[0, 'fft_0_b':'fft_749_b']

plt.figure(figsize=(16, 10))
plt.plot(range(len(sample)), sample)
plt.title("Features fft_0_b through fft_749_b")
plt.show()
```
- **Loading the Data:** The data is loaded from a CSV file named `emotions.csv` into a pandas DataFrame called `data`.
- **Plotting a Sample:** A single sample (first row) from the dataset is selected to visualize its features (`fft_0_b` to `fft_749_b`). These features are likely the Fast Fourier Transform (FFT) coefficients of EEG signals.
- **Visualization:** A plot of these features helps in understanding the pattern and distribution of the data for a single sample.

### 2. **Label Mapping and Preprocessing**

```python
label_mapping = {'NEGATIVE': 0, 'NEUTRAL': 1, 'POSITIVE': 2}

def preprocess_inputs(df):
    df = df.copy()
    df['label'] = df['label'].replace(label_mapping)
    y = df['label'].copy()
    X = df.drop('label', axis=1).copy()
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=123)
    return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = preprocess_inputs(data)
```
- **Label Mapping:** The emotion labels are converted from categorical (text) form to numerical form using a dictionary (`label_mapping`).
- **Preprocessing Function:** 
  - **Label Conversion:** The function `preprocess_inputs` creates a copy of the DataFrame and replaces the text labels with numerical labels.
  - **Feature-Label Split:** The labels (`y`) are separated from the features (`X`).
  - **Train-Test Split:** The data is split into training (70%) and testing (30%) sets to evaluate the model's performance on unseen data.

### 3. **Model Definition**

```python
inputs = tf.keras.Input(shape=(X_train.shape[1],))
expand_dims = tf.expand_dims(inputs, axis=2)
gru = tf.keras.layers.GRU(256, return_sequences=True)(expand_dims)
flatten = tf.keras.layers.Flatten()(gru)
outputs = tf.keras.layers.Dense(3, activation='softmax')(flatten)
model = tf.keras.Model(inputs=inputs, outputs=outputs)
```
- **Input Layer:** Defines the input shape based on the number of features in `X_train`.
- **Expand Dimensions:** The input is expanded to add an additional dimension, which is required for the GRU layer.
- **GRU Layer:** A Gated Recurrent Unit (GRU) layer with 256 units is used. It processes the sequential data and returns the full sequence.
- **Flatten Layer:** The output of the GRU layer is flattened into a single vector.
- **Dense Layer:** A dense (fully connected) layer with 3 units (corresponding to the three emotion classes) and a softmax activation function is used for classification.
- **Model Compilation:** The model is compiled with the Adam optimizer and sparse categorical cross-entropy loss function. The accuracy metric is also specified for evaluation.

### 4. **Model Training**

```python
history = model.fit(
    X_train,
    y_train,
    validation_split=0.2,
    batch_size=32,
    epochs=50,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        )
    ]
)
```
- **Training:** The model is trained on the training data.
  - **Validation Split:** 20% of the training data is used for validation during training.
  - **Batch Size:** The training is done in batches of 32 samples.
  - **Epochs:** The model is trained for up to 50 epochs.
  - **Early Stopping:** Training stops early if the validation loss does not improve for 5 consecutive epochs. The best model weights are restored.

### 5. **Model Evaluation and Saving**

```python
model_acc = model.evaluate(X_test, y_test, verbose=0)[1]
print("Test Accuracy: {:.3f}%".format(model_acc * 100))
model.save("emotion_detection_model")
```
- **Evaluation:** The model is evaluated on the test set, and the test accuracy is printed.
- **Saving:** The trained model is saved to a file named `emotion_detection_model` for later use.

### 6. **Predictions and Confusion Matrix**

```python
y_pred = np.array(list(map(lambda x: np.argmax(x), model.predict(X_test))))

cm = confusion_matrix(y_test, y_pred)
clr = classification_report(y_test, y_pred, target_names=label_mapping.keys())

plt.figure(figsize=(8, 8))
sns.heatmap(cm, annot=True, vmin=0, fmt='g', cbar=False, cmap='Blues')
plt.xticks(np.arange(3) + 0.5, label_mapping.keys())
plt.yticks(np.arange(3) + 0.5, label_mapping.keys())
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

print("Classification Report:\n----------------------\n", clr)
```
- **Predictions:** The model makes predictions on the test set. The `np.argmax` function is used to convert the predicted probabilities to class labels.
- **Confusion Matrix:** The confusion matrix is computed to see how well the model's predictions match the actual labels.
- **Classification Report:** A detailed classification report is generated, showing precision, recall, and F1-score for each class.
- **Visualization:** The confusion matrix is visualized using a heatmap for a clearer understanding of the model's performance.

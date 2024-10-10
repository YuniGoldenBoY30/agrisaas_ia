import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
#from google.colab import drive
from sklearn.model_selection import train_test_split
import os
import json
from PIL import Image
import cv2

# Montar Google Drive
#drive.mount('/content/drive')

# Definir rutas
IMAGE_PATH = 'db_beans_train'
LABEL_PATH = 'db_beans_train'

# Función para cargar y preprocesar imágenes
def load_image(image_path, mask=False):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_png(img, channels=3)
    img = tf.image.resize(img, (256, 256))
    if mask:
        img = tf.image.rgb_to_grayscale(img)
    img = tf.cast(img, tf.float32) / 255.0
    return img

# Función para cargar máscara desde archivo JSON de LabelMe
def load_mask_from_json(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    mask = np.zeros((500, 722), dtype=np.uint8)
    for shape in data['shapes']:
        points = np.array(shape['points'], dtype=np.int32)
        cv2.fillPoly(mask, [points], 1)
    mask = cv2.resize(mask, (256, 256))
    return mask

# Cargar datos
image_paths = [os.path.join(IMAGE_PATH, f) for f in os.listdir(IMAGE_PATH) if f.endswith('.png')]
label_paths = [os.path.join(LABEL_PATH, f.replace('.png', '.json')) for f in os.listdir(IMAGE_PATH) if f.endswith('.png')]

images = [load_image(path) for path in image_paths]
masks = [load_mask_from_json(path) for path in label_paths]

# Dividir datos
X_train, X_val, y_train, y_val = train_test_split(images, masks, test_size=0.2, random_state=42)

# Definir modelo U-Net
def unet_model(input_size=(256, 256, 3)):
    inputs = tf.keras.layers.Input(input_size)
    
    # Encoder (Downsampling)
    conv1 = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')(inputs)
    conv1 = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')(conv1)
    pool1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv1)
    
    conv2 = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same')(pool1)
    conv2 = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same')(conv2)
    pool2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv2)
    
    # Bridge
    conv3 = tf.keras.layers.Conv2D(256, 3, activation='relu', padding='same')(pool2)
    conv3 = tf.keras.layers.Conv2D(256, 3, activation='relu', padding='same')(conv3)
    
    # Decoder (Upsampling)
    up4 = tf.keras.layers.UpSampling2D(size=(2, 2))(conv3)
    up4 = tf.keras.layers.concatenate([up4, conv2])
    conv4 = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same')(up4)
    conv4 = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same')(conv4)
    
    up5 = tf.keras.layers.UpSampling2D(size=(2, 2))(conv4)
    up5 = tf.keras.layers.concatenate([up5, conv1])
    conv5 = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')(up5)
    conv5 = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')(conv5)
    
    outputs = tf.keras.layers.Conv2D(1, 1, activation='sigmoid')(conv5)
    
    model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
    return model

# Compilar modelo
model = unet_model()
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Entrenar modelo
history = model.fit(np.array(X_train), np.array(y_train), validation_data=(np.array(X_val), np.array(y_val)), 
                    epochs=50, batch_size=16)

model.save('unet_model.h5')

# Evaluar y visualizar resultados
loss, accuracy = model.evaluate(np.array(X_val), np.array(y_val))
print(f"Validation Loss: {loss:.4f}")
print(f"Validation Accuracy: {accuracy:.4f}")

# Visualizar una predicción
def visualize_prediction(image, mask, prediction):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    ax1.imshow(image)
    ax1.set_title('Original Image')
    ax2.imshow(mask, cmap='gray')
    ax2.set_title('True Mask')
    ax3.imshow(prediction, cmap='gray')
    ax3.set_title('Predicted Mask')
    plt.show()

sample_image = X_val[0]
sample_mask = y_val[0]
sample_prediction = model.predict(np.expand_dims(sample_image, axis=0))[0,:,:,0]

visualize_prediction(sample_image, sample_mask, sample_prediction)

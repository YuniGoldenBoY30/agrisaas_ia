from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
from PIL import Image
import numpy as np
import io
import base64

app = FastAPI()

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Permite todos los orígenes
    allow_credentials=True,
    allow_methods=["*"],  # Permite todos los métodos
    allow_headers=["*"],  # Permite todos los headers
)

# Cargar el modelo
model = tf.keras.models.load_model('unet_model.h5')

def preprocess_image(image):
    image = image.resize((256, 256))
    image = np.array(image) / 255.0
    return np.expand_dims(image, axis=0)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert('RGB')
    processed_image = preprocess_image(image)
    
    # Realizar la predicción
    prediction = model.predict(processed_image)
    
    # Convertir la predicción en una imagen
    prediction_image = (prediction[0, :, :, 0] * 255).astype(np.uint8)
    prediction_image = Image.fromarray(prediction_image)
    
    # Convertir las imágenes a base64
    buffered = io.BytesIO()
    prediction_image.save(buffered, format="PNG")
    prediction_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
    
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    original_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
    
    return {
        "original_image": original_base64,
        "prediction_image": prediction_base64
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
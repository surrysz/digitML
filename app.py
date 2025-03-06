from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import cv2
from io import BytesIO
from PIL import Image
import tensorflow as tf
from waitress import serve
tf.get_logger().setLevel('ERROR')  # Mostra solo errori e non avvisi

app = Flask(__name__)

# Carica il tuo modello
model = load_model('final_model.h5')

# Funzione per preparare l'immagine per il modello
def prepare_image(image):
    # Se l'immagine non è già un file, convertirla in RGB
    if not isinstance(image, Image.Image):
        image = Image.open(BytesIO(image))
    
    # Converti in RGB (se non lo è già) e ridimensiona a 28x28
    image = image.convert('RGB')
    img = image.resize((28, 28))
    img = img_to_array(img)
    
    # Converti in scala di grigi
    img_gray = 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]
    
    # Aggiungi il canale per (28,28,1)
    img_gray = np.expand_dims(img_gray, axis=-1)
    
    # Inverti i colori se necessario
    if np.mean(img_gray) > 127:
        img_gray = 255 - img_gray
    
    # Reshape e normalizzazione
    img_gray = img_gray.reshape(1, 28, 28, 1)
    img_gray = img_gray.astype('float32') / 255.0
    
    return img_gray

# Endpoint per la root
@app.route('/')
def home():
    return "Flask app is running! Please use the /predict endpoint to upload images."

# Endpoint per ricevere l'immagine
@app.route('/predict', methods=['URL'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'Nessun file caricato'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'File non valido'}), 400

    try:
        # Leggi l'immagine e preparala
        image = Image.open(BytesIO(file.read()))
        img_array = prepare_image(image)
        
        # Fai la predizione
        prediction = model.predict(img_array)
        predicted_class = np.argmax(prediction)
        
        # Risposta JSON con il risultato
        return jsonify({
            'prediction': int(predicted_class),
            'probabilities': prediction.tolist()
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Avvio del server Flask
if __name__ == '__main__':
    import os
    port = int(os.environ.get("PORT", 5000))  # Usa la variabile di ambiente PORT
    serve(app, host="0.0.0.0", port=port)

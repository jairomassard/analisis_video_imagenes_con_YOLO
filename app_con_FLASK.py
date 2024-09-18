from flask import Flask, render_template, request, redirect, url_for
import subprocess
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Definir la carpeta de subida de archivos
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Crear la carpeta de subida si no existe
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect():
    option = request.form['detection_option']
    
    if option == 'camera':
        # Ejecutar detección en la cámara (source 0)
        subprocess.run(['python', 'detect.py', '--source', '0', '--weights', 'yolov5s.pt'])
    else:
        # Procesar el archivo subido
        if 'file' not in request.files:
            return 'No file part'
        
        file = request.files['file']
        
        if file.filename == '':
            return 'No selected file'
        
        if file:
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            
            # Ejecutar detección en video o imagen según la opción
            subprocess.run(['python', 'detect.py', '--source', file_path, '--weights', 'yolov5s.pt'])
    
    return 'Detección completada! Revisa los resultados en el directorio runs.'

if __name__ == '__main__':
    app.run(debug=True)



from flask import Flask, render_template, request
import os
from utils.predict import classify_image

app = Flask(__name__)
UPLOAD_FOLDER = os.path.join('static', 'uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return render_template('index.html', error="No image uploaded")

    file = request.files['image']
    if file.filename == '':
        return render_template('index.html', error="No selected image")

    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(file_path)

    label = classify_image(file_path)
    return render_template('result.html', label=label, image_path=file_path)

if __name__ == '__main__':
    app.run(debug=True)

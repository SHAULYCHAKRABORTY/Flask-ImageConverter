from flask import Flask, render_template, request, send_file
import cv2
import numpy as np
import os

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
PROCESSED_FOLDER = 'processed'

# Ensure directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

def read_img(filepath):
    img = cv2.imread(filepath)
    return img

def edge_detection(img, line_wdt, blur):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    grayBlur = cv2.medianBlur(gray, blur)
    edges = cv2.adaptiveThreshold(
        grayBlur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, line_wdt, blur
    )
    return edges

def color_quantisation(img, k):
    data = np.float32(img).reshape((-1, 3))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.001)
    _, labels, centers = cv2.kmeans(data, k, None, criteria, 10, cv2.KMEANS_PP_CENTERS)
    centers = np.uint8(centers)
    result = centers[labels.flatten()]
    result = result.reshape(img.shape)
    return result

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return "No file uploaded", 400
    file = request.files['file']
    if file.filename == '':
        return "No selected file", 400
    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)
    return render_template('process.html', filename=file.filename)

@app.route('/process', methods=['POST'])
def process():
    transformation = request.form.get('transformation')
    filename = request.form.get('filename')
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    img = read_img(filepath)

    if transformation == 'edge':
        processed_img = edge_detection(img, 9, 7)
    elif transformation == 'cartoon':
        edge_img = edge_detection(img, 9, 7)
        quantised = color_quantisation(img, 4)
        blurred = cv2.bilateralFilter(quantised, d=7, sigmaColor=200, sigmaSpace=200)
        processed_img = cv2.bitwise_and(blurred, blurred, mask=edge_img)
    elif transformation == 'bw':
        processed_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        return "Invalid transformation", 400

    processed_path = os.path.join(PROCESSED_FOLDER, f'processed_{filename}')
    cv2.imwrite(processed_path, processed_img)
    return send_file(processed_path, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)

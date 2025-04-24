from flask import Flask, request, render_template, send_from_directory
from helmet_violations import detect_helmet_violations
import os

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/', methods=['GET', 'POST'])
def index():
    result_text = ""
    annotated_image = None
    if request.method == 'POST':
        if 'image' not in request.files:
            result_text = 'No file part'
        file = request.files['image']
        if file.filename == '':
            result_text = 'No selected file'
        if file:
            filepath = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(filepath)
            result_text, annotated_image = detect_helmet_violations(filepath)
    return render_template('index.html', result=result_text, image_path=annotated_image)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

if __name__ == '__main__':
    app.run(debug=True)

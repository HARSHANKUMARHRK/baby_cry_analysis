from flask import Flask, request, jsonify, render_template
import os
from create_spectogram import create_spectrogram
import threading

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/transform_audio', methods=['POST'])
def transform_audio():

    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    audio_file = request.files['file']
    if audio_file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if audio_file and allowed_file(audio_file.filename):

        audio_file_path = os.path.join('uploads', audio_file.filename)
        audio_file.save(audio_file_path)
        image_file_path = os.path.join('uploads', audio_file.filename.replace('.wav', '.png'))
        thread = threading.Thread(target=create_spectrogram, args=(audio_file_path, image_file_path))
        thread.start()

        return jsonify({'success': True}), 200

    return jsonify({'error': 'Invalid file format'}), 400

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'wav'}

if __name__ == '__main__':
    app.run(debug=True)

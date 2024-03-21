from flask import Flask, send_file, request, jsonify
from flask_cors import CORS  # Import CORS
from PIL import Image
import io
import random
app = Flask(__name__)
import numpy as np 
import librosa
import matplotlib.pyplot as plt
from werkzeug.utils import secure_filename
from pydub import AudioSegment
from io import BytesIO
import matplotlib
import tensorflow as tf
import tempfile


matplotlib.use('agg')
CORS(app)  # Enable CORS for all routes

@app.route('/get-color-image')
def get_color_image():
    print("Random color image requested")

    # Generate a random color
    color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

    # Create an image with RGB mode
    img = Image.new('RGB', (100, 100), color=color)

    # Save the image to a bytes buffer
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    buf.seek(0)

    return send_file(buf, mimetype='image/png')


@app.route('/upload-audio-get-image', methods=['POST'])
def upload_audio_get_image():
    print("Audio received, preparing image")

    # Assuming the audio file doesn't need to be processed for this example
    # In a real application, you might process the audio here

    # Generate a random color image as a response
    color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    img = Image.new('RGB', (100, 100), color=color)
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    buf.seek(0)
    print("returning image")
    return send_file(buf, mimetype='image/png')


def mp3_to_wav(audio_stream):
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as temp_mp3:
        audio_stream.seek(0)
        temp_mp3.write(audio_stream.read())
        temp_mp3.flush()
        audio_segment = AudioSegment.from_file(temp_mp3.name, format="mp3")
        wav_stream = BytesIO()
        audio_segment.export(wav_stream, format="wav")
        wav_stream.seek(0)
    return wav_stream

def mp3tomfcc(audio_stream, max_pad):
    wav_stream = mp3_to_wav(audio_stream)  # Convert MP3 to WAV
    y, sr = librosa.load(wav_stream, sr=None)  # Now load the WAV
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc = np.pad(mfcc, pad_width=((0, 0), (0, max_pad - mfcc.shape[1])), mode='constant')
    return mfcc

@app.route('/upload-audio-get-mfcc-image', methods=['POST'])
def upload_audio_get_mfcc_image():
    # Ensure there's a file part in the request
    if 'file' not in request.files:
        return "No file part", 400
    file = request.files['file']
    if file.filename == '':
        return "No selected file", 400
    if file:
        audio_stream = BytesIO()
        file.save(audio_stream)  # Save the uploaded file to a BytesIO object
        
        # Convert the BytesIO stream to MFCC
        mfcc = mp3tomfcc(audio_stream, max_pad=400)

        # Create an image from the MFCC
        plt.figure(figsize=(10, 4))  # Adjusted figsize for better visibility
        plt.imshow(mfcc, aspect='auto', origin='lower')
        buf = BytesIO()
        plt.savefig(buf, format='PNG')
        buf.seek(0)
        plt.close()

        return send_file(buf, mimetype='image/png')

@app.route('/classify-tone', methods=['POST'])
def classify_tone_dummy():
    possible_classifications = ["First Tone", "Second Tone", "Third Tone", "Fourth Tone", "Neutral"]
    return random.choice(possible_classifications)

# @app.route('/classify-tone', methods=['POST'])

# # model = tf.keras.models.load_model('path/to/your/model.h5')

# def classify_tone():
#     # Get the audio file from the request
#     audio_file = request.files['file']

#     # Preprocess the audio file
#     processed_audio = mp3tomfcc(audio_file)

#     # Make a prediction
#     prediction = model.predict(np.array([processed_audio]))[0]
#     predicted_class = np.argmax(prediction) + 1  # Assuming classes are 1-indexed

#     # Map prediction to tones
#     tone_mapping = {1: "First Tone", 2: "Second Tone", 3: "Third Tone", 4: "Fourth Tone", 5: "Neutral"}
#     tone = tone_mapping.get(predicted_class, "Unknown")

#     return jsonify({"classification": tone})


if __name__ == '__main__':
    app.run(debug=True, port = 5000)
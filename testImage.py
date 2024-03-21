import cv2
import numpy as np 
import librosa
import matplotlib.pyplot as plt

def upload_audio_get_mfcc_image():
    print("Audio received, converting to MFCC")

    # Read the audio file from the request
    audio_file = request.files['file']
    audio_path = audio_file.filename

    # Convert to MFCC
    mfcc = mp3tomfcc(audio_path, max_pad=400)

    # Create an image from the MFCC
    plt.figure(figsize=(10, 4))
    plt.imshow(mfcc, aspect='auto', origin='lower')
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)

    return send_file(buf, mimetype='image/png')

def mp3tomfcc(file_path, max_pad):
    audio, sample_rate = librosa.load(file_path)
    mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=60)
    pad_width = max_pad - mfcc.shape[1]
    mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
    return mfcc
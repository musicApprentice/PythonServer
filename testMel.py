import librosa
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import io

def mfcc_to_image(mfcc_data):
    # Create a figure and axis to plot the heatmap
    fig, ax = plt.subplots()
    # Plot the heatmap
    img = ax.imshow(mfcc_data, interpolation='nearest', cmap='hot', aspect='auto')
    # Turn off the axis labels
    ax.axis('off')
    # Save the image to a buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    # Close the figure to free memory
    plt.close(fig)
    print("Reached the end of image conversion with no errors")
    return buf

def mp3tomfcc(file_path, max_pad):
    print("starting conversion mp3 to mfcc")
    audio, sample_rate = librosa.load(file_path)
    mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=60)
    pad_width = max_pad - mfcc.shape[1]
    mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
    print("reached the end of mp3 to mfcc")

    return mfcc

# Test the function with an MP3 file



mfcc_result = mp3tomfcc('test_recording.mp3', 400)  # Use your MP3 file and max_pad value
print("MFCC Shape:", mfcc_result.shape)

mfcc_image_buf = mfcc_to_image(mfcc_result)


# To display the image, uncomment the following lines
from IPython.display import display, Image
display(Image(data=mfcc_image_buf.getvalue()))
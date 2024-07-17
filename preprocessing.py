import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

# Function to create spectrogram from .wav file and save as an image
def create_spectrogram(file_name, output_image):
    y, sr = librosa.load(file_name)
    plt.figure(figsize=(2.56, 2.56))
    S = librosa.feature.melspectrogram(y, sr=sr)
    S_dB = librosa.power_to_db(S, ref=np.max)
    librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel')
    plt.axis('off')
    plt.savefig(output_image, bbox_inches='tight', pad_inches=0)
    plt.close()
    img = Image.open(output_image).convert("RGB")
    img = img.resize((256, 256), Image.ANTIALIAS)
    img.save(output_image)

# Paths
wav_dir = "/Users/sashwathselvakumar/Documents/dataset/Cough sounds in patients with pulmonary tuberculosis"
spectrogram_dir = "/Users/your_username/Desktop/audio_project/TBSpect"
os.makedirs(spectrogram_dir, exist_ok=True)

# Load labels from a CSV file
labels_df = pd.read_csv('/Users/your_username/Desktop/audio_project/labels.csv')

# Create a DataFrame for spectrogram images
spectrogram_labels = []

# Walk through all subdirectories and files
for root, dirs, files in os.walk(wav_dir):
    for wav_file in files:
        if wav_file.endswith('.wav'):
            file_path = os.path.join(root, wav_file)
            base_name = os.path.splitext(wav_file)[0]
            label = labels_df[labels_df['filename'] == f"{base_name}.wav"]['label'].values[0]
            for i in range(226):
                output_image = os.path.join(spectrogram_dir, f"{base_name}_{i}.png")
                create_spectrogram(file_path, output_image)
                spectrogram_labels.append({'filename': f"{base_name}_{i}.png", 'label': label})

# Convert to DataFrame and save to CSV
spectrogram_labels_df = pd.DataFrame(spectrogram_labels)
spectrogram_labels_df.to_csv('/Users/your_username/Desktop/audio_project/experiment2.csv', index=False, header=False)

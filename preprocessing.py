import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

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

wav_dir = 'path/to/your/wav/files'
spectrogram_dir = 'path/to/save/spectrograms'
os.makedirs(spectrogram_dir, exist_ok=True)

wav_files = [f for f in os.listdir(wav_dir) if f.endswith('.wav')]

for wav_file in wav_files:
    file_path = os.path.join(wav_dir, wav_file)
    output_image = os.path.join(spectrogram_dir, f"{os.path.splitext(wav_file)[0]}.png")
    create_spectrogram(file_path, output_image)

labels_df = pd.read_csv('path/to/labels.csv')

spectrogram_files = [f for f in os.listdir(spectrogram_dir) if f.endswith('.png')]
spectrogram_labels = []

for spect_file in spectrogram_files:
    base_name = os.path.splitext(spect_file)[0]
    label = labels_df[labels_df['filename'] == f"{base_name}.wav"]['label'].values[0]
    spectrogram_labels.append({'filename': spect_file, 'label': label})

spectrogram_labels_df = pd.DataFrame(spectrogram_labels)
spectrogram_labels_df.to_csv('experiment2.csv', index=False, header=False)

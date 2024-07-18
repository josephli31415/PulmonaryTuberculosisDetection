import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

def create_spectrogram(file_name, output_image):
    y, sr = librosa.load(file_name)
    plt.figure(figsize=(2.56, 2.56))
    S = librosa.feature.melspectrogram(y=y, sr=sr)
    S_dB = librosa.power_to_db(S, ref=np.max)
    librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel')
    plt.axis('off')
    plt.savefig(output_image, bbox_inches='tight', pad_inches=0)
    plt.close()
    img = Image.open(output_image).convert("RGB")
    img = img.resize((256, 256), Image.LANCZOS)
    img.save(output_image)

# Define paths
wav_dir = "/Users/sashwathselvakumar/Documents/dataset/Cough sounds in patients with pulmonary tuberculosis"
spectrogram_dir = "/Users/sashwathselvakumar/Documents/dataset/TBSpect"
os.makedirs(spectrogram_dir, exist_ok=True)

labels_df = pd.read_csv('/Users/sashwathselvakumar/Documents/dataset/labels.csv')

print("Column names in the CSV file:", labels_df.columns)

spectrogram_labels = []

for root, dirs, files in os.walk(wav_dir):
    for wav_file in files:
        if wav_file.endswith('.wav'):
            file_path = os.path.join(root, wav_file)
            base_name = os.path.splitext(wav_file)[0]
            constructed_file_name = f"{base_name}.wav"
            print(f"Looking for label for file: {constructed_file_name}")
            if constructed_file_name in labels_df['Filename'].values:
                label = labels_df[labels_df['Filename'] == constructed_file_name]['Label'].values[0]
                output_image = os.path.join(spectrogram_dir, f"{base_name}.png")
                create_spectrogram(file_path, output_image)
                spectrogram_labels.append({'filename': f"{base_name}.png", 'label': label})
            else:
                print(f"File {constructed_file_name} not found in labels.csv")

spectrogram_labels_df = pd.DataFrame(spectrogram_labels)
spectrogram_labels_df.to_csv('/Users/sashwathselvakumar/Documents/dataset/experiment2.csv', index=False, header=False)

print("Conversion completed and CSV saved.")

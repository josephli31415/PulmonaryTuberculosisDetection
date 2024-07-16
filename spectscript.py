import os
import librosa
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


import pandas as pd
import numpy as np

csv_file_path = 'path/to/your/file.csv'
df = pd.read_csv(csv_file_path)
data_array = df.to_numpy()
npy_file_path = 'path/to/save/experiment2.npy'
np.save(npy_file_path, data_array)

print(f"Data saved to {npy_file_path}")


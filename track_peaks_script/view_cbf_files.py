import fabio
import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np

Z_DIR = "/mnt/z/VEGA/CsV3Sb5_strain/2024/07/CsV3Sb5_July24_Kalpha/runs"
T = "80K"
V = "V16.0_-16.0"
if T == "80K":
    peak_dir = "6peaks"
elif T == "15K":
    peak_dir = "7peaks"

run, frame = "02", "022"

path = os.path.join(Z_DIR, T, V, peak_dir, "merged/", f"CsV3Sb5_strain_merged_{run}_{frame}.cbf")

img = fabio.open(path)
data = img.data
# Save the data to a CSV file
output_file = "intensity_data.csv"
np.savetxt(output_file, data, delimiter=",")
plt.imshow(data, cmap='seismic')
plt.colorbar()
plt.show()
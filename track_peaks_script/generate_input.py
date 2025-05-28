import pandas as pd
import numpy as np
from itertools import product

# Automate the generation based on step size and maximum value
step = 0.25
max_value = 3.00

# Define allowed values based on the given constraint
# h and k must be either integer or 0.5, l can include 0.25 and 0.75 as well
hk_values = hk_values = np.round(np.arange(0, max_value+step*2, step*2), 2)
l_values = np.round(np.arange(0, max_value+step, step), 2)

# Generate valid combinations where h and k come from hk_values, and l from l_values
combinations = list(product(hk_values, hk_values, l_values))

# Create DataFrame
df_auto = pd.DataFrame(combinations, columns=["h", "k", "l"])

# Save to CSV
csv_path_auto = "input_find_peaks.csv"
df_auto.to_csv(csv_path_auto, index=False)

#!/usr/bin/env python3
import ast
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

# To run the code do, for example: python3 plot_hk_data.py ./data_to_plot/hk_data__Region1_03l.csv --temp 80 --v_eff 16 --plane "(h,k,-0.5)"

# Custom parser for temperatures like "80K"
def parse_temp(s):
    """
    Strip trailing 'K' or 'k' and whitespace, then convert to float.
    e.g. '80K' -> 80.0
    """
    return float(s.strip().rstrip('K').rstrip('k'))

# 1) Load data with all necessary converters
def load_data(fn):
    df = pd.read_csv(
        fn,
        sep='\t',
        converters={
            'Intensity Map Data (np.array)': ast.literal_eval,
            'Plot limits':                ast.literal_eval,
            'Norm Input':                 ast.literal_eval,
            'Temperature (K)':             parse_temp,
        }
    )

    # 2) Create an "effective" voltage column (post-doubling)
    df['V_eff (V)'] = df['Voltage (V)'] * 2.0

    # 3) Clean up plane strings
    df['Reciprocal Space Plane'] = df['Reciprocal Space Plane'].str.strip()

    return df

# 4) Plotting function: pick one map by T, V_eff, plane
def plot_intensity(df, temp, v_eff, plane):
    # Filter for the desired row
    mask = (
        (df['Temperature (K)']        == temp) &
        (df['V_eff (V)']              == v_eff) &
        (df['Reciprocal Space Plane'] == plane.strip())
    )
    df_sel = df[mask]

    if df_sel.empty:
        print("No matching row found. Available values are:")
        print(" Temperatures:", sorted(df['Temperature (K)'].unique()))
        print(" Voltages (eff):", sorted(df['V_eff (V)'].unique()))
        print(" Planes:", sorted(df['Reciprocal Space Plane'].unique()))
        raise SystemExit(1)

    if len(df_sel) > 1:
        print(f"Warning: found {len(df_sel)} matches; using the first one.")

    row = df_sel.iloc[0]

    matrix = np.array(row['Intensity Map Data (np.array)'])
    extent = row['Plot limits']    # (xmin, xmax, ymin, ymax)
    vmin, vmax = row['Norm Input'] # (vmin, vmax)

    fig, ax = plt.subplots(figsize=(7,7))
    norm = Normalize(vmin=vmin, vmax=vmax)
    img  = ax.imshow(
        matrix,
        cmap="terrain_r",
        norm=norm,
        origin='lower',
        extent=extent
    )
    plt.colorbar(img, ax=ax, label='Intensity')
    ax.set_xlabel('X [a.u.]')
    ax.set_ylabel('Y [a.u.]')
    ax.set_title(f"T={temp} K, V={v_eff} V, plane={plane}")
    plt.tight_layout()
    plt.show()

# 5) CLI entry point
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot one intensity map by T, V_eff, and plane"
    )
    parser.add_argument("csvfile", help="Path to your tab-separated CSV file.")
    parser.add_argument(
        "--temp",    type=float, required=True,
        help="Temperature in Kelvin (e.g. 80)"
    )
    parser.add_argument(
        "--v_eff",   type=float, required=True,
        help="Effective voltage after doubling (V)"
    )
    parser.add_argument(
        "--plane",   type=str,   required=True,
        help="Reciprocal space plane, e.g. '(h,k,0)'"
    )
    args = parser.parse_args()

    df = load_data(args.csvfile)
    plot_intensity(df, args.temp, args.v_eff, args.plane)

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
from read_reciprocal_space_cuts import check_existing_data, extract_plane_from_filename, reorder_files_to_data, check_merged#, generate_pattern

def generate_data(data, Planes_list, is_merged, temperature, voltage, local_path, ratio, N_pixel):
    # Initial Parameters
    zone_1 = (840, 1000, 990, 1130)
    zone_2 = (840, 1000, 990, 1130)   # (col_start, col_end, row_start, row_end)/(X_start, X_end, Y_start, Y_end); defines the boxe delimitation for the hk space of comparison
    zone_3 = (840, 1000, 990, 1130)

    peak_BoxSize = (7, 7)

    peaks_positions_1 = {0: (891,1005), 1: (860, 1058), 2: (891, 1112),
                       3: (952,1112), 4: (984, 1059), 5: (953, 1005)} # as (collum, row)/(X, Y)
    peaks_positions_2 = {0: (891,1005), 1: (860, 1058), 2: (891, 1112),
                       3: (952,1112), 4: (984, 1059), 5: (953, 1005)} # as (collum, row)/(X, Y)
    peaks_positions_3 = {0: (891,1005), 1: (860, 1058), 2: (891, 1112),
                       3: (952,1112), 4: (984, 1059), 5: (953, 1005)} # as (collum, row)/(X, Y)

    # Processing
    list_peaks_positions = [peaks_positions_1, peaks_positions_2, peaks_positions_3]
    for peaks_positions in list_peaks_positions:
        zone = list_peaks_positions.index(peaks_positions) + 1
        PeaksPositions_RangeForCalc = {}
        index = 0
        for peak_pos in peaks_positions.values():
            x, y = peak_pos[0], peak_pos[1]
            delta_x, delta_y = peak_BoxSize[0], peak_BoxSize[1]
            PeaksPositions_RangeForCalc[index] = (y - delta_y, y + delta_y, x - delta_x, x + delta_x)
            index += 1

        for plane, plane_data in zip(Planes, data):
            index = 0
            peaks_intensities = []
            for inputs in PeaksPositions_RangeForCalc.values():
                peak_data = plane_data[inputs[0]: inputs[1], inputs[2]: inputs[3]]
                data_intensity = np.mean(peak_data)
                peaks_intensities.append(round(float(data_intensity),2))
                index += 1
            
            # Calculate average intensities of symmetric pairs
            I_2 = (peaks_intensities[0] + peaks_intensities[3]) / 2   # Domain 2
            I_1 = (peaks_intensities[1] + peaks_intensities[4]) / 2   # Domain 1
            I_3 = (peaks_intensities[2] + peaks_intensities[5]) / 2  # Domain 3
            return [voltage, zone, plane, I_1, I_2, I_3]

def get_data(original_path, Planes_list, temperature, ratio, N_pixel):
    os.chdir(original_path)
    data_dataframe = []
    for voltage in os.listdir():
        local_path = f"./{voltage}/data"
        os.chdir(local_path)
        gather_needed = check_existing_data(local_path, Planes_list)
        if not gather_needed:
            img_files = [os.path.join(local_path, f) for f in os.listdir(local_path) 
                            if f.endswith(".img") and extract_plane_from_filename(f) in Planes_list]
            if img_files:
                data = reorder_files_to_data(img_files, Planes_list)
                print('Processing existing files')
                is_merged = check_merged(img_files[0])
                line = generate_data(data, Planes_list, is_merged, temperature, voltage, local_path, ratio, N_pixel).copy()
                data_dataframe.append(line)
        os.chdir(original_path)
    return data_dataframe

def process_data(data_file, out_file_name):
    # Read the input data
    df = pd.read_csv(data_file, sep='\t')

    # Group by Voltage and Plane, and average I_1, I_2, I_3 across Data_Zone
    grouped = df.groupby(['Voltage', 'Plane'])[['I_1', 'I_2', 'I_3']].mean().reset_index()

    # Calculate I_total = I_1 + I_2 + I_3
    grouped['I_total'] = grouped['I_1'] + grouped['I_2'] + grouped['I_3']

    # Calculate normalized intensities
    grouped['I_1(normalised)'] = grouped['I_1'] / grouped['I_total']
    grouped['I_2(normalised)'] = grouped['I_2'] / grouped['I_total']
    grouped['I_3(normalised)'] = grouped['I_3'] / grouped['I_total']

    # Calculate I_av = (I_1(normalised) + I_2(normalised) + I_3(normalised)) / 3
    grouped['I_av'] = (grouped['I_1(normalised)'] + grouped['I_2(normalised)'] + grouped['I_3(normalised)']) / 3

    # Calculate Anisotropy Parameter A
    # A = sqrt[(I_1 - I_av)^2 + (I_2 - I_av)^2 + (I_3 - I_av)^2] / I_av
    grouped['A'] = np.sqrt(
        (grouped['I_1(normalised)'] - grouped['I_av'])**2 +
        (grouped['I_2(normalised)'] - grouped['I_av'])**2 +
        (grouped['I_3(normalised)'] - grouped['I_av'])**2
    ) / grouped['I_av']

    # Save to a new .data file
    grouped.to_csv(out_file_name, sep='\t', index=False, float_format='%.2f')
    print("Treated data saved to:", out_file_name)
    return None

def plot_data(processed_data_file, temperature):
    # Read the processed data
    df = pd.read_csv(processed_data_file, sep='\t')

    # Map numerical plane values to Miller plane strings
    plane_mapping = {0.0: '(h,k,0)', -0.25: '(h,k,-0.25)', -0.5: '(h,k,-0.5)'}
    df['Plane'] = df['Plane'].astype(float).map(plane_mapping)

    # Validate temperature
    if temperature not in ['80K', '15K']:
        raise ValueError("Temperature must be '80K' or '15K'")

    # Set seaborn style for better visuals
    sns.set(style="whitegrid")

    # Define colors for I_1, I_2, I_3
    colors = {'I_1(normalised)': 'blue', 'I_2(normalised)': 'green', 'I_3(normalised)': 'red'}
    # Slightly altered colors for Voltage = 0.0 at 80K
    colors_special = {'I_1(normalised)': 'lightblue', 'I_2(normalised)': 'lightgreen', 'I_3(normalised)': 'lightcoral'}

    # Determine planes for joint and separate plots based on temperature
    if temperature == '80K':
        joint_planes = ['(h,k,-0.25)', '(h,k,-0.5)']
        separate_plane = '(h,k,0)'
    else:  # 15K
        joint_planes = ['(h,k,0)', '(h,k,-0.5)']
        separate_plane = '(h,k,-0.25)'

    # 1. Plot normalized intensities
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    fig.suptitle(f'Normalized Intensities vs. Voltage ({temperature})', fontsize=16)

    for idx, plane in enumerate(joint_planes):
        ax = axes[idx]
        plane_data = df[df['Plane'] == plane]
        
        for intensity in ['I_1(normalised)', 'I_2(normalised)', 'I_3(normalised)']:
            if temperature == '80K':
                # Split data for Voltage = 0.0 and others
                zero_data = plane_data[plane_data['Voltage'] == 0.0]
                non_zero_data = plane_data[plane_data['Voltage'] != 0.0]
                if not zero_data.empty:
                    ax.scatter(zero_data['Voltage'], zero_data[intensity], 
                              c=colors_special[intensity], label=f'{intensity} (V=0)', s=50, marker='o')
                if not non_zero_data.empty:
                    ax.scatter(non_zero_data['Voltage'], non_zero_data[intensity], 
                              c=colors[intensity], label=intensity, s=50, marker='o')
            else:
                # No special treatment for 15K
                ax.scatter(plane_data['Voltage'], plane_data[intensity], 
                          c=colors[intensity], label=intensity, s=50, marker='o')
        
        ax.set_title(f'Plane {plane}')
        ax.set_xlabel('Voltage')
        ax.set_ylabel('Normalized Intensity')
        ax.legend()

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(os.path.join(os.path.dirname(processed_data_file), f'intensities_joint_{temperature}.png'))
    plt.close()

    # Separate plot for the remaining plane
    fig, ax = plt.subplots(figsize=(6, 5))
    plane_data = df[df['Plane'] == separate_plane]
    
    for intensity in ['I_1(normalised)', 'I_2(normalised)', 'I_3(normalised)']:
        if temperature == '80K':
            zero_data = plane_data[plane_data['Voltage'] == 0.0]
            non_zero_data = plane_data[plane_data['Voltage'] != 0.0]
            if not zero_data.empty:
                ax.scatter(zero_data['Voltage'], zero_data[intensity], 
                          c=colors_special[intensity], label=f'{intensity} (V=0)', s=50, marker='o')
            if not non_zero_data.empty:
                ax.scatter(non_zero_data['Voltage'], non_zero_data[intensity], 
                          c=colors[intensity], label=intensity, s=50, marker='o')
        else:
            ax.scatter(plane_data['Voltage'], plane_data[intensity], 
                      c=colors[intensity], label=intensity, s=50, marker='o')
    
    ax.set_title(f'Normalized Intensities vs. Voltage ({separate_plane}, {temperature})')
    ax.set_xlabel('Voltage')
    ax.set_ylabel('Normalized Intensity')
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(processed_data_file), f'intensities_separate_{temperature}.png'))
    plt.close()

    # 2. Plot Anisotropy Parameter (A)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    fig.suptitle(f'Anisotropy Parameter vs. Voltage ({temperature})', fontsize=16)

    for idx, plane in enumerate(joint_planes):
        ax = axes[idx]
        plane_data = df[df['Plane'] == plane]
        
        if temperature == '80K':
            zero_data = plane_data[plane_data['Voltage'] == 0.0]
            non_zero_data = plane_data[plane_data['Voltage'] != 0.0]
            if not zero_data.empty:
                ax.scatter(zero_data['Voltage'], zero_data['A'], 
                          c='orange', label='A (V=0)', s=50, marker='o')
            if not non_zero_data.empty:
                ax.scatter(non_zero_data['Voltage'], non_zero_data['A'], 
                          c='darkorange', label='A', s=50, marker='o')
        else:
            ax.scatter(plane_data['Voltage'], plane_data['A'], 
                      c='darkorange', label='A', s=50, marker='o')
        
        ax.set_title(f'Plane {plane}')
        ax.set_xlabel('Voltage')
        ax.set_ylabel('Anisotropy Parameter (A)')
        ax.legend()

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(os.path.join(os.path.dirname(processed_data_file), f'anisotropy_joint_{temperature}.png'))
    plt.close()

    # Separate A plot for the remaining plane
    fig, ax = plt.subplots(figsize=(6, 5))
    plane_data = df[df['Plane'] == separate_plane]
    
    if temperature == '80K':
        zero_data = plane_data[plane_data['Voltage'] == 0.0]
        non_zero_data = plane_data[plane_data['Voltage'] != 0.0]
        if not zero_data.empty:
            ax.scatter(zero_data['Voltage'], zero_data['A'], 
                      c='orange', label='A (V=0)', s=50, marker='o')
        if not non_zero_data.empty:
            ax.scatter(non_zero_data['Voltage'], non_zero_data['A'], 
                      c='darkorange', label='A', s=50, marker='o')
    else:
        ax.scatter(plane_data['Voltage'], plane_data['A'], 
                  c='darkorange', label='A', s=50, marker='o')
    
    ax.set_title(f'Anisotropy Parameter vs. Voltage ({separate_plane}, {temperature})')
    ax.set_xlabel('Voltage')
    ax.set_ylabel('Anisotropy Parameter (A)')
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(processed_data_file), f'anisotropy_separate_{temperature}.png'))
    plt.close()

    # 3. Plot I_total and I_av averaged across all planes
    # Group by Voltage and average I_total and I_av
    avg_data = df.groupby('Voltage')[['I_total', 'I_av']].mean().reset_index()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True)
    fig.suptitle(f'I_total and I_av vs. Voltage (Averaged Across Planes, {temperature})', fontsize=16)

    # I_total plot
    axes[0].scatter(avg_data['Voltage'], avg_data['I_total'], c='purple', s=50, marker='o')
    axes[0].set_title('I_total (Averaged)')
    axes[0].set_xlabel('Voltage')
    axes[0].set_ylabel('I_total')

    # I_av plot
    axes[1].scatter(avg_data['Voltage'], avg_data['I_av'], c='teal', s=50, marker='o')
    axes[1].set_title('I_av (Averaged)')
    axes[1].set_xlabel('Voltage')
    axes[1].set_ylabel('I_av')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(os.path.join(os.path.dirname(processed_data_file), f'total_av_{temperature}.png'))
    plt.close()

    print(f"Plots saved for {temperature}:")
    print(f"- Intensities (joint): intensities_joint_{temperature}.png")
    print(f"- Intensities (separate): intensities_separate_{temperature}.png")
    print(f"- Anisotropy (joint): anisotropy_joint_{temperature}.png")
    print(f"- Anisotropy (separate): anisotropy_separate_{temperature}.png")
    print(f"- I_total and I_av: total_av_{temperature}.png")
    return None

running_script_path = os.getcwd()

T = "80K"
main_path = f"./Data/{T}/"
Planes = ["(h,k,0)", "(h,k,-0.25)", "(h,k,-0.5)"]
ratio_Pixels_to_iAngs, Number_pixels = 0.00813008, N_pixel = 1476

data_df = get_data(main_path, Planes, T, ratio_Pixels_to_iAngs, Number_pixels)

os.chdir(running_script_path)
# Create DataFrame
df = pd.DataFrame(data_df, columns=['Voltage', 'Data_Zone', 'Plane', 'I_1', 'I_2', 'I_3'])
# Save to a .data file (tab-separated text file)
df.to_csv(f'{T}_hk.data', sep='\t', index=False, float_format='%.2f')

input_data_file = f'{T}_hk.data'
output_file_name = f'{T}_hk_output.data'

if input_data_file:
    process_data(input_data_file, output_file_name)

if output_file_name:
    plot_data(output_file_name, T)















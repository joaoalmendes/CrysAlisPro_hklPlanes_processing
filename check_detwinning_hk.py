import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import re
import shutil
from read_reciprocal_space_cuts import check_existing_data, extract_plane_from_filename, reorder_files_to_data, check_merged, find_correct_data_path

def generate_data(data, Planes_list, merged, voltage):
    # Initial Parameters
    peak_BoxSize = (7, 7)

    peaks_positions_1 = {0: (522,1005), 1: (491, 1058), 2: (523, 1112),
                       3: (583,1112), 4: (615, 1059), 5: (584, 1005)} # as (collum, row)/(X, Y)
    peaks_positions_2 = {0: (891,1005), 1: (860, 1058), 2: (891, 1112),
                       3: (952,1112), 4: (984, 1059), 5: (953, 1005)} # as (collum, row)/(X, Y)
    peaks_positions_3 = {0: (1079,685), 1: (1048, 738), 2: (1079, 792),
                       3: (1140,792), 4: (1172, 738), 5: (1141, 685)} # as (collum, row)/(X, Y)

    # Processing
    list_peaks_positions = [peaks_positions_1, peaks_positions_2, peaks_positions_3]
    result = []  # Collect all rows
    for peaks_positions in list_peaks_positions:
        zone = list_peaks_positions.index(peaks_positions) + 1
        PeaksPositions_RangeForCalc = {}
        index = 0
        for peak_pos in peaks_positions.values():
            x, y = peak_pos[0], peak_pos[1]
            delta_x, delta_y = peak_BoxSize[0], peak_BoxSize[1]
            PeaksPositions_RangeForCalc[index] = (y - delta_y, y + delta_y, x - delta_x, x + delta_x)
            index += 1

        for plane, plane_data in zip(Planes_list, data):
            index = 0
            peaks_intensities = []
            for inputs in PeaksPositions_RangeForCalc.values():
                peak_data = plane_data[inputs[0]: inputs[1], inputs[2]: inputs[3]]
                if merged == True:
                    data_intensity = np.mean(peak_data) / 2
                else:
                    data_intensity = np.mean(peak_data)
                peaks_intensities.append(round(float(data_intensity),2))
                index += 1
            
            # Calculate average intensities of symmetric pairs
            I_2 = (peaks_intensities[0] + peaks_intensities[3]) / 2   # Domain 2
            I_1 = (peaks_intensities[1] + peaks_intensities[4]) / 2   # Domain 1
            I_3 = (peaks_intensities[2] + peaks_intensities[5]) / 2   # Domain 3
            print(f"Appending row: {[voltage, zone, plane, I_1, I_2, I_3]}")
            result.append([voltage, zone, plane, I_1, I_2, I_3])
    return result

def get_data(original_path, Planes_list, temperature, not_to_process_voltages):
    VEGA_dir = "/mnt/z/VEGA/CsV3Sb5_strain/2024/07/CsV3Sb5_July24_Kalpha/runs"
    os.chdir(original_path)
    data_dataframe = []

    # Get list of folders in the current directory
    current_voltages = [f for f in os.listdir() if os.path.isdir(f)]

    # Get list of folders starting with 'V' in the VEGA_dir/temperature directory
    v_folders = [f for f in os.listdir(os.path.join(VEGA_dir, f"{temperature}")) 
                 if os.path.isdir(os.path.join(VEGA_dir, f"{temperature}", f)) and f.startswith('V')]

    # Extract voltage values from v_folders for comparison (e.g., 'V8.0_-8.0' -> '8.0')
    v_folders_voltages = []
    for folder in v_folders:
        match = re.match(r'V([\d.]+)_-[\d.]+', folder)
        if match:
            v_folders_voltages.append(match.group(1))  # Extract the voltage (e.g., '8.0')

    # Compare current_voltages with v_folders_voltages
    if set(current_voltages) != set(f"{v}V" for v in v_folders_voltages):
        # Create missing folders in the current directory
        for voltage in v_folders_voltages:
            folder_name = f"{float(voltage)}V"  # Format as '8.0V'
            folder_path = os.path.join(original_path, folder_name)
            data_folder_path = os.path.join(folder_path, "data")
            
            # Create the voltage folder and 'data' subfolder if they don't exist
            os.makedirs(folder_path, exist_ok=True)
            os.makedirs(data_folder_path, exist_ok=True)

    # Filter out voltages to skip
    process_voltages = [v for v in current_voltages if v not in not_to_process_voltages]

    os.chdir(original_path)
    for voltage in process_voltages:
        # Use os.path.join for proper path construction
        local_path = os.path.join(original_path, voltage, "data")
        
        # Ensure the data folder exists
        os.makedirs(local_path, exist_ok=True)
        os.chdir(local_path)
        
        gather_needed = check_existing_data(local_path, Planes_list)
        if not gather_needed:
            img_files = [os.path.join(local_path, f) for f in os.listdir(local_path) 
                         if f.endswith(".img") and extract_plane_from_filename(f) in Planes_list]
            if img_files:
                data = reorder_files_to_data(img_files, Planes_list)
                print(f'Processing existing files for {voltage}')
                is_merged = check_merged(img_files[0])
                # Remove 'V' from voltage for float conversion
                voltage_value = float(voltage.replace('V', ''))
                lines = generate_data(data, Planes_list, is_merged, voltage_value).copy()
                data_dataframe.extend(lines)
        else:
            # Determine the correct remote path dynamically
            voltage_value = voltage.replace('V', '')  # Extract voltage (e.g., '8.0' from '8.0V')
            remote_base_path = os.path.join(VEGA_dir, str(temperature), f"V{voltage_value}_-{voltage_value}")
            remote_data_path = find_correct_data_path(remote_base_path)
            if remote_data_path is None:
                print(f"No valid data folder found in base directory: {remote_base_path}")
                # Remove the local voltage folder and its contents
                local_voltage_folder = os.path.join(original_path, voltage)
                if os.path.exists(local_voltage_folder):
                    shutil.rmtree(local_voltage_folder)
                    print(f"Removed local folder: {local_voltage_folder}")
                continue

            # Process and copy required files only if missing or being replaced
            found_files = 0
            print(f"Processing {voltage}")

            for file_name in os.listdir(remote_data_path):
                plane = extract_plane_from_filename(file_name)
                if plane and plane in Planes_list:
                    found_files += 1
                    src_file = os.path.join(remote_data_path, file_name)

                    if not os.path.exists(src_file):
                        print(f"Plane {plane} not found in {temperature}/{voltage} at {remote_data_path}")
                    else:
                        dest_file = os.path.join(local_path, file_name)
                        os.makedirs(os.path.dirname(dest_file), exist_ok=True)
                        shutil.copy(src_file, dest_file)

                        # Handle `.Identifier` files
                        identifier_file = os.path.join(local_path, f"{os.path.splitext(file_name)[0]}.Identifier")
                        if os.path.exists(identifier_file):
                            os.remove(identifier_file)

            if found_files != len(Planes_list):
                print(f"Expected {len(Planes_list)} files but found {found_files} for planes in {temperature}/{voltage} at {remote_data_path}")

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

    # Convert Voltage column to float, removing 'V' suffix if present
    df['Voltage'] = df['Voltage'].astype(str).str.replace('V', '').astype(float)

    # Validate temperature
    if temperature not in ['80K', '15K']:
        raise ValueError("Temperature must be '80K' or '15K'")

    # Set seaborn style for better visuals
    sns.set(style="whitegrid")

    # Define colors for I_1, I_2, I_3
    colors = {'I_1(normalised)': 'blue', 'I_2(normalised)': 'green', 'I_3(normalised)': 'red'}
    colors_special = {'I_1(normalised)': 'lightblue', 'I_2(normalised)': 'lightgreen', 'I_3(normalised)': 'lightcoral'}

    # Determine planes for joint and separate plots based on temperature
    if temperature == '80K':
        joint_planes = ['(h,k,-0.25)', '(h,k,-0.5)']
        separate_plane = '(h,k,0)'
    else:  # 15K
        joint_planes = ['(h,k,0)', '(h,k,-0.5)']
        separate_plane = '(h,k,-0.25)'

    # 1. Plot normalized intensities (joint plot with shared legend)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    fig.suptitle(f'Normalized Intensities vs. Voltage ({temperature})', fontsize=16)
    
    # Collect legend handles and labels
    handles, labels = [], []
    
    for idx, plane in enumerate(joint_planes):
        ax = axes[idx]
        plane_data = df[df['Plane'] == plane]
        
        for intensity in ['I_1(normalised)', 'I_2(normalised)', 'I_3(normalised)']:
            if temperature == '80K':
                # Split data for Voltage = 0.0 and others
                zero_data = plane_data[plane_data['Voltage'] == 0.0]
                non_zero_data = plane_data[plane_data['Voltage'] != 0.0]
                if not zero_data.empty:
                    scatter = ax.scatter(zero_data['Voltage'], zero_data[intensity], 
                                        c=colors_special[intensity], label=f'{intensity[:3]} (V=0)', s=50, marker='o')
                    if idx == 0:
                        handles.append(scatter)
                        labels.append(f'{intensity[:3]} (V=0)')
                if not non_zero_data.empty:
                    scatter = ax.scatter(non_zero_data['Voltage'], non_zero_data[intensity], 
                                        c=colors[intensity], label=intensity[:3], s=50, marker='o')
                    if idx == 0:
                        handles.append(scatter)
                        labels.append(intensity[:3])
            else:
                scatter = ax.scatter(plane_data['Voltage'], plane_data[intensity], 
                                    c=colors[intensity], label=intensity[:3], s=50, marker='o')
                if idx == 0:
                    handles.append(scatter)
                    labels.append(intensity[:3])
        
        ax.set_title(f'Plane {plane}')
        ax.set_xlabel('Voltage (V)')
        ax.set_ylabel('Normalized Intensity')

    # Debugging: Print handles and labels
    print(f"Intensities Joint Plot ({temperature}) - Legend Labels: {labels}")
    
    # Add shared legend outside the plot
    if handles:
        fig.legend(handles, labels, loc='center right', bbox_to_anchor=(0.99, 0.5), 
                   borderaxespad=0., fontsize='small')
        fig.subplots_adjust(right=0.90)  # Reserve less space for legend
    else:
        print(f"Warning: No legend handles for intensities joint plot ({temperature})")

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
                          c=colors_special[intensity], label=f'{intensity[:3]} (V=0)', s=50, marker='o')
            if not non_zero_data.empty:
                ax.scatter(non_zero_data['Voltage'], non_zero_data[intensity], 
                          c=colors[intensity], label=intensity[:3], s=50, marker='o')
        else:
            ax.scatter(plane_data['Voltage'], plane_data[intensity], 
                      c=colors[intensity], label=intensity[:3], s=50, marker='o')
    
    ax.set_title(f'Normalized Intensities vs. Voltage ({separate_plane}, {temperature})')
    ax.set_xlabel('Voltage (V)')
    ax.set_ylabel('Normalized Intensity')
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(processed_data_file), f'intensities_separate_{temperature}.png'))
    plt.close()

    # 2. Plot Anisotropy Parameter (joint plot with shared legend)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    fig.suptitle(f'Anisotropy Parameter vs. Voltage ({temperature})', fontsize=16)
    
    handles, labels = [], []
    
    for idx, plane in enumerate(joint_planes):
        ax = axes[idx]
        plane_data = df[df['Plane'] == plane]
        
        if temperature == '80K':
            zero_data = plane_data[plane_data['Voltage'] == 0.0]
            non_zero_data = plane_data[plane_data['Voltage'] != 0.0]
            if not zero_data.empty:
                scatter = ax.scatter(zero_data['Voltage'], zero_data['A'], 
                                    c='orange', label='A (V=0)', s=50, marker='o')
                if idx == 0:
                    handles.append(scatter)
                    labels.append('A (V=0)')
            if not non_zero_data.empty:
                scatter = ax.scatter(non_zero_data['Voltage'], non_zero_data['A'], 
                                    c='darkorange', label='A', s=50, marker='o')
                if idx == 0:
                    handles.append(scatter)
                    labels.append('A')
        else:
            scatter = ax.scatter(plane_data['Voltage'], plane_data['A'], 
                                c='darkorange', label='A', s=50, marker='o')
            if idx == 0:
                handles.append(scatter)
                labels.append('A')
        
        ax.set_title(f'Plane {plane}')
        ax.set_xlabel('Voltage (V)')
        ax.set_ylabel('Anisotropy Parameter (A)')

    # Debugging: Print handles and labels
    print(f"Anisotropy Joint Plot ({temperature}) - Legend Labels: {labels}")
    
    # Add shared legend outside the plot
    if handles:
        fig.legend(handles, labels, loc='center right', bbox_to_anchor=(0.99, 0.5), 
                   borderaxespad=0., fontsize='small')
        fig.subplots_adjust(right=0.90)  # Reserve less space for legend
    else:
        print(f"Warning: No legend handles for anisotropy joint plot ({temperature})")

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
    ax.set_xlabel('Voltage (V)')
    ax.set_ylabel('Anisotropy Parameter (A)')
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(processed_data_file), f'anisotropy_separate_{temperature}.png'))
    plt.close()

    # 3. Plot I_total and I_av averaged across all planes
    avg_data = df.groupby('Voltage')[['I_total', 'I_av']].mean().reset_index()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True)
    fig.suptitle(f'I_total and I_av vs. Voltage (Averaged Across Planes, {temperature})', fontsize=16)

    axes[0].scatter(avg_data['Voltage'], avg_data['I_total'], c='purple', s=50, marker='o')
    axes[0].set_title('I_total (Averaged)')
    axes[0].set_xlabel('Voltage (V)')
    axes[0].set_ylabel('I_total')

    axes[1].scatter(avg_data['Voltage'], avg_data['I_av'], c='teal', s=50, marker='o')
    axes[1].set_title('I_av (Averaged)')
    axes[1].set_xlabel('Voltage (V)')
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
voltages_skip_list = ["48.0V"]  # For 80K
#voltages_skip_list = []  # For 15K
main_path = running_script_path + f"/Data/{T}/"
Planes = ["(h,k,0)", "(h,k,-0.25)", "(h,k,-0.5)"]

data_df = get_data(main_path, Planes, T, voltages_skip_list)

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















import fabio    # to read the .img files
from silx.gui.plot import Plot2D
import PyQt5.QtCore  # Importing PyQt5 will force silx to use it
from silx.gui import qt
from silx.gui.colors import Colormap
from PyQt5.QtCore import QRectF
from PyQt5.QtGui import QPen, QColor
from PyQt5.QtWidgets import QGraphicsEllipseItem
import os
import re
import shutil
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np
from matplotlib.colors import Normalize
from matplotlib.colorbar import ColorbarBase
import random
import pandas as pd
from scipy.signal import find_peaks
from intensity_hkPlanes import intensity_hk_plane, check_merged

# Log peak data to the log.data file
def log_data(message: str, info: str, log_file="log.data") -> None:
    with open(log_file, "a") as f:  # Append to log file
        f.write(message + "\n")
        f.write(info + "\n")
        f.write("\n")

# Logging function for errors during code execution
def log_error(message: str, log_file="log.error") -> None:
    print(message)  # Print to console
    with open(log_file, "a") as f:  # Append to log file
        f.write(message + "\n")
        f.write("\n")

# Visualize using silx
def visualize(img_file: np.array) -> None:
    # Initialize the Qt application
    app = qt.QApplication([])

    # Create a 2D plot
    plot = Plot2D()
    plot.setWindowTitle(f"CrystAlis {img_file} data")

    # Create a Colormap object; here minimum value is 0 and maximum 100 but for different data (sets) this might need adjustment
    # But adjust as you think it's more sutiable to you or your data
    # Colormaps: viridis, gray_r, seismic
    max_cap = 100
    colormap = Colormap(name="gray_r", vmin=0, vmax=max_cap, normalization="linear")

    # Add the image to the plot with the Colormap object
    plot.addImage(img_file, colormap=colormap, legend="Image Data")
    plot.show()

    # Start the Qt event loop
    app.exec_()

    # Close and clean up the application
    plot.close()
    del plot
    qt.QApplication.quit()  # Ensure the application is properly shut down

def reorder_files_to_data(img_files, Planes):
    # Reorder data to align with planes
    # Map extracted plane names to corresponding files
    plane_to_file = {extract_plane_from_filename(os.path.basename(f)): f for f in img_files}
    # Sort files according to the order in PLANES
    sorted_files = [plane_to_file[plane] for plane in Planes if plane in plane_to_file]
    # Read the files with Fabio and pass them in np.array format to be precessed
    data = [fabio.open(file).data for file in sorted_files]
    return data

def extract_plane_from_filename(filename: str) -> str | None:
    # Try matching "strain_" pattern first (with optional "merged_")
    match = re.search(r'strain_(merged_)?([^\.]+(?:\.\d+)?)(?:\.img)?', filename)
    if match:
        extracted = match.group(2)  # Get the part after "strain_" or "strain_merged_"
        extracted = extracted.replace("_", ",")  # Replace underscores with commas
        return f"({extracted})"
    
    # Try matching "O120_twin3_" pattern
    match = re.search(r'O120_twin3_([^\.]+(?:\.\d+)?)(?:\.img)?', filename)
    if match:
        extracted = match.group(1)  # Get the part after "O120_twin3_"
        extracted = extracted.replace("_", ",")  # Replace underscores with commas
        return f"({extracted})"
    
    print(f"Warning: Could not determine plane for filename: {filename}")
    return None

def process_img_files(data: np.array, output_dir: str, temperature: str, voltage: str, Planes: list[str], merged: bool, dataframe: list) -> None:
    # Defines a starting and ending collumn for the exctraction of the peak intensity data
    def calculate_columns(center, width = 10):
        start = center - width // 2
        end = center + width // 2 + 1
        return start, end
    
    # Filters out fake very intense peaks from the data if encontered
    def filter_large_values(data: list[np.ndarray], multiplier=10) -> tuple[np.ndarray, float]:
        while True:
            if data.size <= 1:
                # Not enough data points to filter
                break

            # Calculate the average
            avg = np.mean(data)

            # Define the threshold
            threshold = multiplier * avg

            # Identify indices of values exceeding the threshold
            large_indices = np.where(data > threshold)[0]

            if large_indices.size == 0:
                # No values exceed the threshold; break the loop
                break

            # Remove the values exceeding the threshold
            data = np.delete(data, large_indices)

        # Calculate the average of the filtered data
        filtered_avg = np.mean(data) if data.size > 0 else 0
        return data, filtered_avg
    
    def peak_finding(data: list[np.ndarray], height=None, min_prominence=18, min_distance=5) -> tuple[np.ndarray, int, float]:
        # Ensure data is a numpy array
        data = np.asarray(data)

        # Identify candidate peaks: points higher than their neighbors
        greater_than_prev = np.r_[False, data[1:] > data[:-1]]  # Left condition
        greater_than_next = np.r_[data[:-1] > data[1:], False]  # Right condition
        peak_candidates = np.where(greater_than_prev & greater_than_next)[0]

        # Apply prominence filter
        prominences = data[peak_candidates]
        prominent_peaks = peak_candidates[prominences >= min_prominence]

        # Enforce minimum distance between peaks
        if min_distance > 1:
            filtered_peaks = []
            last_peak = -min_distance  # Start with an invalid peak
            for peak in prominent_peaks:
                if peak - last_peak >= min_distance:
                    filtered_peaks.append(peak)
                    last_peak = peak
            prominent_peaks = np.array(filtered_peaks)
        filtered_peaks, avg_peak_I = filter_large_values(prominent_peaks)
        N_peaks = len(filtered_peaks) 
        return filtered_peaks, N_peaks, avg_peak_I

    def detect_peaks_scipy(L_values: np.ndarray, intensity: np.ndarray):
        """
        Detects peaks in intensity data and returns their positions and maximum intensities.

        Parameters:
        L_values (np.ndarray): 1D array of L positions.
        intensity (np.ndarray): 1D array of intensity values.
        height (float, optional): Minimum height of peaks.
        distance (int, optional): Minimum number of points between peaks.
        prominence (float, optional): Minimum prominence of peaks.

        Returns:
        tuple: (peak_positions, peak_intensities), both as numpy arrays.
        """
        # Find peaks in the intensity data
        height, prominence = np.mean(intensity)+ 0.5 * np.std(intensity), 0.1 * (np.max(intensity) - np.min(intensity))  
        peak_indices, properties = find_peaks(intensity, height=height, threshold=height, distance=None, prominence=prominence, width=None, wlen=None, rel_height=0.5, plateau_size=None)
        
        # Get the corresponding L positions and intensity values of peaks
        peak_positions = L_values[peak_indices]
        peak_intensities = intensity[peak_indices]

        return peak_positions, peak_intensities

    def initial_parameters(plane: str) -> list[tuple, tuple]:
            # This is going to change from case to case, that's why first one does a visual analysis
            params = {
                "(h,3,l)": [(245, 369, 676, 738), (369, 490, 676, 738)], # this is as (X0, X1, Y0, Y1)
                "(3,k,l)": [(619, 739, 738, 801), (739, 862, 738, 801)],
                "(1-h,2+k,l)": [(739, 862, 675, 738), (862, 986, 675, 738)],
                "(0,k,l)": [(985, 985, 674, 737), (985, 985, 674, 737)],    # this is an example if one wants to scan the BPs in-plane position along l
                "(h,0,l)": [(987, 987, 738, 801), (987, 987, 738, 801)],
                "(h,1-k,l)": [(863, 863, 801, 865), (863, 863, 801, 865)],
                #"(h,3,l)": [(615, 615, 610, 675), (615, 615, 610, 675)],   # For the new weaker peaks
            }
            return params.get(plane, [])

    log_data(f"Temperature: {temperature}, Voltage: {voltage}", "")

    planes_plot_data = [[] for _ in range(len(Planes))]  # Ensure separate lists for each plane

    # Run the data processing for each plane
    Y_width = 10
    noise_cap = 100
    plane_idx = 0
    for plane, plane_data in zip(Planes, data):
        #N_pixel = np.sqrt(np.size(data)/2)  # If one does not know the pixel data size

        # Visualize the data previously if needed/wanted, at first this is all one needs to do in order to extract the information needed to run the code systematically
        #visualize(plane_data)

        par_list = initial_parameters(plane)
        log_data(f"Plane: {plane}", "")
        peak_index = 0
        for par in par_list:
            X0, X1, Y0, Y1 = par
            X_start, X_end = calculate_columns(int((X0+X1)/2))
            #Y0, Y1 = Y0-Y_width, Y1+Y_width
            Y0, Y1 = Y0+int(Y_width/2) + 5, Y1-int(Y_width/2) - 5  # If excluding BPs is necessary
            roi_data = plane_data[Y0:Y1, X_start:X_end]
            row_means = np.mean(roi_data, axis=1)   # Calculate the mean across each row for the given the column values
            
            # Get the peak's data and save them in the log.data file
            plot_data = filter_large_values(row_means, multiplier=30)[0]
            size = np.size(plot_data)

            y_0, y_1 = Y0 - N_pixel/2, Y1 - N_pixel/2 # rescale the origin for the peaks 

            plot_data[plot_data > noise_cap] = 0.0     # If removing BPs intensity is needed

            l_plot = ratio * np.linspace(y_0, y_1, size)
            if merged == True:             # Handle the merged data to keep the results consistent
                I_plot = plot_data / 2
            else:
                I_plot = plot_data

            planes_plot_data[plane_idx].append((l_plot, I_plot))
            dataframe.append([temperature, float(voltage[:-1]), plane, l_plot.tolist(), I_plot.tolist()])   # Weak peaks
            #dataframe.append([float(temperature[:-1]), float(voltage[:-1]), plane, l_plot.tolist(), I_plot.tolist()])   # Intensity(l) plots

            peaks_pos, N_peaks, avg_peak_I = peak_finding(I_plot)
            sp_peaks_pos, sp_peaks_I = detect_peaks_scipy(l_plot, I_plot)
            log_data(f"Peak {peak_index}", f"Peaks: {[round(float(p+y_0)*ratio,2) for p in peaks_pos]}, Average peak intensity: {round(avg_peak_I,2)}")
            log_data(f"Peak {peak_index}", f"Peaks: {[round(float(p+y_0)*ratio,2) for p in sp_peaks_pos.tolist()]}")


            peak_index += 1
        plane_idx += 1
    
    # Create DataFrame with columns for main data and arrays
    df = pd.DataFrame(dataframe, columns=['Temperature (K)', 'Voltage (V)', 'Reciprocal Space Plane', 'l (np.array)', 'Intensity (np.array)'])
    # Save to a single tab-separated CSV file
    df.to_csv('data_to_plot/weakPeaks_data.csv', sep='\t', index=False)
    
    return planes_plot_data

def plots_function(plot_dict, Planes, T, voltages=None, mode='overlay', save_fig=False, save_path='plot.png'):
    """
    Generates plots for intensity vs. L values from given voltage-dependent data.

    Parameters:
    plot_dict: dict
        Dictionary where keys are voltage values (as strings) and values are lists of lists.
        Each inner list corresponds to an HKL plane and contains two tuples of np.arrays.
    voltages: list, optional
        List of specific voltages to plot. If None, uses all available voltages.
    mode: str
        'side_by_side' -> Separate figures for each voltage with two subplots for comparison.
        'overlay' -> Overlays intensity plots for different voltages.
        'stacked' -> Stacked evolution plots for multiple voltages with a side voltage axis.
    save_fig: bool, optional
        Whether to save the figure instead of just displaying it.
    save_path: str, optional
        File path to save the figure if save_fig is True.
    """

    log_data(f"Plot Mode: {mode}", "")

    if voltages is None:
        voltages = list(plot_dict.keys())  # Use all available voltages
    
    num_planes = len(Planes)  # Determine number of HKL planes
    # Colormaps: tab20, tab10, nipy_spectral, plasma, viridis
    colors = plt.cm.viridis(np.linspace(0, 1, len(voltages)))  # Color gradient for voltages
    
    for plane_idx in range(num_planes):
        
        if mode == 'side_by_side':
            for voltage in voltages:
                fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
                plane_data = plot_dict[voltage][plane_idx]
                for pos in range(2):
                    L_plot, I_plot = plane_data[pos]
                    axes[pos].plot(L_plot, I_plot, label=f'Peak {pos+1}', color='b' if pos == 0 else 'r')
                    axes[pos].set_ylabel("Intensity")
                    axes[pos].legend()
                    axes[pos].set_title(f"T={T}, V={voltage} - HKL Plane {Planes[plane_idx]} - Peak {pos+1}")
                
                axes[-1].set_xlabel("L")
                plt.tight_layout()
                
                if save_fig:
                    plt.savefig(f"{save_path}_plane{plane_idx+1}_V{voltage}.png", dpi=300)
                    plt.close()
                else:
                    plt.show()
                    plt.close()
        
        elif mode == 'overlay':
            fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
            
            for v_idx, voltage in enumerate(voltages):
                plane_data = plot_dict[voltage][plane_idx]
                for pos in range(2):
                    L_plot, I_plot = plane_data[pos]
                    axes[pos].plot(L_plot, I_plot, label=f'V={voltage}', color=colors[v_idx])
                    axes[pos].legend()
                    axes[pos].set_ylabel("Intensity")
                    axes[pos].set_title(f"HKL Plane {Planes[plane_idx]} - Peak {pos+1}")
                    
            axes[-1].set_xlabel("L")
            plt.tight_layout()
            
            if save_fig:
                plt.savefig(f"{save_path}_plane{plane_idx+1}_overlay.png", dpi=300)
                plt.close()
            else:
                plt.show()
                plt.close()
        
        elif mode == 'stacked':
            for pos in range(2):
                fig, ax = plt.subplots(figsize=(6, 8))
                
                max_I = 0
                shift = 0
                for v_idx, voltage in enumerate(voltages):
                    plane_data = plot_dict[voltage][plane_idx]
                    L_plot, I_plot = plane_data[pos]
                    shift += max_I * 1.2  # Stack plots with spacing

                    ax.text(L_plot[-1] + 0.1 * (max(L_plot) - min(L_plot)),  # a bit to the right
                                        min(I_plot) + shift,
                                        f"{voltage}",
                                        va='center', ha='left',
                                        fontsize=12, color='black')

                    ax.plot(L_plot, I_plot + shift, color='k')
                    max_I = max(I_plot)
                
                ax.set_xlabel("L")
                ax.set_ylabel("Intensity")
                ax.set_title(f"HKL Plane {Planes[plane_idx]} - Peak {pos+1}")
                plt.tight_layout()
                
                if save_fig:
                    plt.savefig(f"{save_path}_plane{plane_idx+1}_stacked_pos{pos+1}.png", dpi=300)
                    plt.close()
                else:
                    plt.show()
                    plt.close()

def extract_temp_voltage(path: str) -> tuple[str, str]:
    # Assumes structure: <base_dir>/<temperature>/<voltage>/data
    print(path)
    parts = path.split(os.sep)
    print(parts)
    temperature = parts[-3]
    voltage = parts[-2]
    return temperature, voltage

def generate_pattern(plane: str, merged) -> None:
    """
    Generates the expected .img file name based on the plane input.
    Uses the exact function format you specified.
    """
    prefix = "CsV3Sb5_strain_merged_" if merged else "CsV3Sb5_strain_"
    if plane.startswith("(") and plane.endswith(")"):
        plane_content = plane.strip("()").replace(",", "_")
        return f"{prefix}{plane_content}.img"
    return None

def check_existing_data(local_data_path: str, Planes: list[str]) -> bool:
    """
    Checks if all requested plane .img files exist in the local data folder.
    If they do, prompts the user to decide whether to replace them.

    Args:
        local_data_path (str): Path to the local data directory.
        Planes (list[str]): List of planes to check.

    Returns:
        bool: True if data gathering should proceed, False if skipped.
    """
    if not os.path.exists(local_data_path):
        return True  # Proceed if local data folder doesn't exist

    existing_files = set(os.listdir(local_data_path))

    # If no files exist in the directory, proceed with data gathering
    if not existing_files:
        print(f"No files found in {local_data_path}. Proceeding with data gathering.")
        return True

    # Auto-detect merged flag from an existing file
    sample_file = random.choice(list(existing_files))
    merged = "merged" in sample_file.lower()

    expected_files = {generate_pattern(plane, merged) for plane in Planes}

    missing_files = expected_files - existing_files  # Files that are missing
    all_exist = len(missing_files) == 0

    if all_exist:
        user_input = input(f"All requested files are already present in {local_data_path}. Replace? (y/n): ").strip().lower()
        return user_input == "y"

    if missing_files:
        print(f"Missing files detected in {local_data_path}: {missing_files}. Proceeding with data gathering.")

    return True  # Continue processing

def find_correct_data_path(base_path):
    """
    Determines the correct path for retrieving .img files based on the directory structure.
    
    Parameters:
    base_path (str): Path to the temperature/voltage folder in the cloud storage.
    
    Returns:
    str: Correct directory path containing the .img files, or None if not found.
    """
    fulltth_extended_path = os.path.join(base_path, "fulltth_extended", "esp", "unwarp")
    fulltth_path = os.path.join(base_path, "fulltth", "merged", "unwarp")
    
    if os.path.exists(fulltth_extended_path):
        return fulltth_extended_path
    elif os.path.exists(fulltth_path):
        return fulltth_path
    else:
        return None

def process_data(base_dir: str, local_dir: str, Planes: list[str], TEMPERATURES: list[str], VOLTAGES: dict[str, list[str]], ratio: float, N_pixel: int, processing_mode: str) -> None:
    """
    Main function that checks for existing data, processes new data if needed, and organizes files.

    Args:
        base_dir (str): Base directory containing temperature-voltage data.
        local_dir (str): Local directory to store gathered data.
        Planes (list[str]): List of crystallographic planes to process.
        TEMPERATURES (list[str]): List of temperatures to process.
        VOLTAGES (dict[str, list[str]]): Dictionary mapping temperatures to voltages.
    """
    data_dir = os.path.join(local_dir, "Data")  # Check only local storage first
    dataframe = []
    for temperature in TEMPERATURES:
        temp_path = os.path.join(data_dir, temperature)
        if not os.path.isdir(temp_path):
            continue

        plot_dic ={}
        for voltage in VOLTAGES.get(temperature, []):
            voltage_path = os.path.join(temp_path, voltage)
            local_data_path = os.path.join(voltage_path, "data")

            # Check if gathering is needed before accessing the remote base_dir
            gather_needed = check_existing_data(local_data_path, Planes)
            if not gather_needed:
                print(f"Skipping data gathering for {temperature}, {voltage}.")
                # Still process existing files even if no gathering is needed
                img_files = [os.path.join(local_data_path, f) for f in os.listdir(local_data_path) 
                           if f.endswith(".img") and extract_plane_from_filename(f) in Planes]
                if img_files:
                    data = reorder_files_to_data(img_files, Planes)
                    print('Processing existing files')
                    is_merged = check_merged(img_files[0])
                    if processing_mode == "hk_planes":
                        intensity_hk_plane(data, Planes, is_merged, temperature, voltage, local_dir, ratio, N_pixel, dataframe)
                    else:
                        plot_dic[voltage] = process_img_files(data, voltage_path, temperature, voltage, Planes, is_merged, dataframe)
                continue

            # Determine the correct remote path dynamically
            remote_base_path = os.path.join(base_dir, temperature, f"V{voltage[:-1]}_-{voltage[:-1]}")
            remote_data_path = find_correct_data_path(remote_base_path)
            if remote_data_path is None:
                log_error(f"No valid data folder found in base directory: {remote_data_path}")
                continue

            # Process and copy required files only if missing or being replaced
            found_files = 0
            print(f"{voltage}")

            for file_name in os.listdir(remote_data_path):
                plane = extract_plane_from_filename(file_name)
                if plane and plane in Planes:
                    found_files += 1
                    src_file = os.path.join(remote_data_path, file_name)

                    if os.path.exists(src_file) == False:
                        log_error(f"Plane {plane} not found in {temperature}/{voltage} at {remote_data_path}")
                    else:
                        dest_file = os.path.join(local_data_path, file_name)
                        os.makedirs(os.path.dirname(dest_file), exist_ok=True)
                        shutil.copy(src_file, dest_file)

                        # Handle `.Identifier` files
                        identifier_file = os.path.join(local_data_path, f"{os.path.splitext(file_name)[0]}.Identifier")
                        if os.path.exists(identifier_file):
                            os.remove(identifier_file)

            if found_files != len(Planes):
                print(f"No matching files found for planes in {temperature}/{voltage} at {remote_data_path}")

            # Collect and process img files after copying
            img_files = [os.path.join(local_data_path, f) for f in os.listdir(local_data_path) 
                        if f.endswith(".img") and extract_plane_from_filename(f) in Planes]
            if img_files:
                    data = reorder_files_to_data(img_files, Planes)
                    print('Processing existing files')
                    is_merged = check_merged(img_files[0])
                    if processing_mode == "hk_planes":
                        intensity_hk_plane(data, Planes, is_merged, temperature, voltage, local_dir, ratio, N_pixel, dataframe)
                    else:
                        plot_dic[voltage] = process_img_files(data, voltage_path, temperature, voltage, Planes, is_merged, dataframe)
        if processing_mode == "hk_planes":
            continue
        else:
            pass
            #plots_function(plot_dic, Planes, temperature)
    return dataframe
# Inputs and code execution order

# Inputs: Define temperatures and voltages to process
TEMPERATURES = [
                "80K", 
                "15K",
                #"80K_medium_strain", 
                #"15K_medium_strain",
                #"75K",
                ]  # Add temperatures here

VOLTAGES = {
            #"80K": [ "0.0V", "8.0V", "12.0V", "14.5V", "20.0V", "29.0V", "38.0V", "55.0V"],
            #"15K": ["5.0V", "20.0V", "35.0V", "57.0V", "95.0V", "125.0V"],
            "80K": ["12.0V", "20.0V", "29.0V"], 
            "15K": ["5.0V", "57.0V", "125.0V"],
            #"80K_medium_strain": ["30.0V"], 
            #"15K_medium_strain": ["83.0V"],
            #"75K": ["58.0V"], 
            }  # Voltages for each temperature

# Define the planes to be processed with regards to your inputed parameters in the processing functions
#PLANES = ["(h,3,l)", "(3,k,l)", "(1-h,2+k,l)"]     # For Intensity(l) plots of l dependent RSCs
PLANES = ["(h,k,0)", "(h,k,-0.25)", "(h,k,-0.5)"]   # For hk intensity plots
#PLANES = ["(0,k,l)", "(h,0,l)"]     # For Intensity(l) plots to observe the weaker (new) peaks observed

ratio = 0.01578947 #(l per pixel); Ratio to convert pixel units to l units calculated from gathered visual data where one concludes that 190 pixels correspond to 3l
ratio_hkPlanes = 0.00813008
N_pixel = 1476

# Code execution order
# Possible modes
# 'None'-> Ivsl plots; 'hk_planes' -> hk plane intensity color map

if __name__ == "__main__":
    # Base directory where the temperature folders are located, in the Cloud Storage
    base_dir = "/mnt/z/VEGA/CsV3Sb5_strain/2024/07/CsV3Sb5_July24_Kalpha/runs"
    local_dir = os.getcwd() 
    dataframe = process_data(base_dir, local_dir, PLANES, TEMPERATURES, VOLTAGES, ratio = ratio_hkPlanes, N_pixel = N_pixel, processing_mode="hk_planes")
    if True:    # For hk planes
        # Create DataFrame with columns for main data and arrays
        df = pd.DataFrame(dataframe, columns=['Temperature (K)', 'Voltage (V)', 'Reciprocal Space Plane', 'Intensity Map Data (np.array)'])
        import json
        # Convert arrays to JSON strings
        df['Intensity Map Data (np.array)'] = df['Intensity Map Data (np.array)'].apply(json.dumps)
        # Save to a single tab-separated CSV file
        df.to_csv('data_to_plot/hk_data__Region1_03l.csv', sep='\t', index=False)












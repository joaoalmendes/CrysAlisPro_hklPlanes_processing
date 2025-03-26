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
    max_cap = 100
    colormap = Colormap(name="viridis", vmin=0, vmax=max_cap, normalization="linear")

    # Add the image to the plot with the Colormap object
    plot.addImage(img_file, colormap=colormap, legend="Image Data")
    plot.show()

    # Start the Qt event loop
    app.exec_()

    # Close and clean up the application
    plot.close()
    del plot
    qt.QApplication.quit()  # Ensure the application is properly shut down

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

# Reorders the planes data gathering order with our input plane order so there are no miss placements while robustly handling missing planes during reordering
def reorder_data(img_files: list[str], planes: list[str]) -> tuple[list[np.ndarray], list[bool], list[float]]:
    file_planes = []
    is_merged = []
    l_list = []

    for f in img_files:
        plane = extract_plane_from_filename(os.path.basename(f))
        l = plane.strip("()").split(",")[2]
        if plane is None:
            print(f"Warning: Could not determine plane for file {f}")
        file_planes.append(plane)
        l_list.append(l)  # Store the value of l for later use in the visualization step
        is_merged.append("merged" in os.path.basename(f))

    if None in file_planes:
        log_error(f"Some files do not have recognizable plane information in the dataset.")
        return [], []  # Gracefully return empty lists to avoid halting

    ordered_data = []
    ordered_is_merged = []
    ordered_l_list = []
    for plane in planes:
        try:
            index = file_planes.index(plane)  # Find the index of the plane in file_planes
            ordered_data.append(fabio.open(img_files[index]).data)
            ordered_l_list.append(l_list[index])
            ordered_is_merged.append(is_merged[index])
        except ValueError:
            continue  # Skip missing planes and move on
    return ordered_data, ordered_is_merged, ordered_l_list

def process_img_files(img_files: list[str], output_dir: str, temperature: str, voltage: str, Planes: list[str]) -> None:

    # Reorder data to align with planes
    data, is_merged, l_values = reorder_data(img_files, Planes)
    
    #N_pixel = np.sqrt(np.size(data)/2)  # If one doen't know the pixel data size

    # Visualize the data previoustly if needed/wanted, at first this is all one needs to do in order to extract the information needed to run the code systematically
    #visualize(data[0])

    log_data("Running for Intensity(l) plots.", "\n")
    log_data(f"Temperature: {temperature}, Voltage: {voltage}", "\n")

    def initial_parameters(plane: str) -> list[tuple, tuple]:
        # This is going to change from case to case, that's why first one does a visual analysis, this data (parameters) is taken from the ROI.dat file and inputed here
        params = {
            "(h,3,l)": [(245, 369, 676, 738), (369, 490, 676, 738)],
            "(3,k,l)": [(619, 739, 738, 801), (739, 863, 738, 801)],
        }
        return params.get(plane, [])
    
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
    
    def find_peaks(data: list[np.ndarray], height=None, min_prominence=20, min_distance=5) -> tuple[np.ndarray, int, float]:
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

    planes_plot_data = [[]] * len(Planes)

    # Run the data processing for each plane
    Y_width = 5
    noise_cap = 10
    plane_idx = 0
    for plane, plane_data in zip(Planes, data):
        par_list = initial_parameters(plane)
        log_data(f"Plane: {plane}", "\n")
        peak_index = 0
        for par in par_list:
            X0, X1, Y0, Y1 = par
            X_start, X_end = calculate_columns(int((X0+X1)/2))
            roi_data = plane_data[Y0-Y_width:Y1+Y_width, X_start:X_end]
            row_means = np.mean(roi_data, axis=1)   # Calculate the mean across each row for the given the column values
            
            # Get the peak's data and save them in the log.data file
            find_peaks_data = find_peaks(row_means)
            row_means_filtered = filter_large_values(row_means, multiplier=30)[0]
            row_means_filtered[row_means_filtered < noise_cap] = 0.0
            size = np.size(row_means_filtered)
            y_0, y_1 = Y0 - N_pixel/2, Y1 - N_pixel/2 # rescale the origin for the peaks 

            log_data(f"Peak {peak_index}", f"Peaks: {[round(float(p+y_0)*ratio,2) for p in find_peaks_data[0]]}, Average peak intensity: {round(find_peaks_data[2],2)}")

            l_plot = ratio * np.linspace(y_0, y_1, size)
            I_plot = row_means_filtered

            planes_plot_data[plane_idx].append((l_plot, I_plot))

            peak_index += 1
        plane_idx += 1

    return planes_plot_data

def plots_function(plot_dict, voltages=None, mode='stacked', save_fig=False, save_path='plot.png'):
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
    if voltages is None:
        voltages = list(plot_dict.keys())  # Use all available voltages
    
    num_planes = len(next(iter(plot_dict.values())))  # Determine number of HKL planes
    colors = plt.cm.viridis(np.linspace(0, 1, len(voltages)))  # Color gradient for voltages
    
    for plane_idx in range(num_planes):
        
        if mode == 'side_by_side':
            for voltage in voltages:
                fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
                plane_data = plot_dict[voltage][plane_idx]
                
                for pos in range(2):
                    L_plot, I_plot = plane_data[pos]
                    axes[pos].plot(L_plot, I_plot, label=f'Position {pos+1}', color='b' if pos == 0 else 'r')
                    axes[pos].set_ylabel("Intensity")
                    axes[pos].legend()
                    axes[pos].set_title(f"V={voltage} - HKL Plane {plane_idx+1} - Position {pos+1}")
                
                axes[-1].set_xlabel("L")
                plt.tight_layout()
                
                if save_fig:
                    plt.savefig(f"{save_path}_plane{plane_idx+1}_V{voltage}.png", dpi=300)
                else:
                    plt.show()
        
        elif mode == 'overlay':
            fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
            
            for v_idx, voltage in enumerate(voltages):
                plane_data = plot_dict[voltage][plane_idx]
                
                for pos in range(2):
                    L_plot, I_plot = plane_data[pos]
                    axes[pos].plot(L_plot, I_plot, label=f'V={voltage}', color=colors[v_idx])
                    axes[pos].legend()
                    axes[pos].set_ylabel("Intensity")
                    axes[pos].set_title(f"Overlay - HKL Plane {plane_idx+1} - Position {pos+1}")
                    
            axes[-1].set_xlabel("L")
            plt.tight_layout()
            
            if save_fig:
                plt.savefig(f"{save_path}_plane{plane_idx+1}_overlay.png", dpi=300)
            else:
                plt.show()
        
        elif mode == 'stacked':
            for pos in range(2):
                fig, ax = plt.subplots(figsize=(8, 6))
                
                shifts = []
                for v_idx, voltage in enumerate(voltages):
                    plane_data = plot_dict[voltage][plane_idx]
                    L_plot, I_plot = plane_data[pos]
                    shift = v_idx * max(I_plot) * 1.2  # Stack plots with spacing
                    shifts.append(shift)
                    ax.plot(L_plot, I_plot + shift, color='k')
                
                # Create a secondary y-axis for voltage labels
                ax2 = ax.twinx()
                ax2.set_yticks(shifts)
                ax2.set_yticklabels([f'{v}' for v in voltages], fontsize=12, color='black')
                ax2.set_ylabel("Voltage", fontsize=14, color='black')
                
                ax.set_xlabel("L")
                ax.set_ylabel("Intensity")
                ax.set_title(f"Stacked - HKL Plane {plane_idx+1} - Position {pos+1}")
                plt.tight_layout()
                
                if save_fig:
                    plt.savefig(f"{save_path}_plane{plane_idx+1}_stacked_pos{pos+1}.png", dpi=300)
                else:
                    plt.show()

def extract_temp_voltage(path: str) -> tuple[str, str]:
    # Assumes structure: <base_dir>/<temperature>/<voltage>/data
    print(path)
    parts = path.split(os.sep)
    print(parts)
    temperature = parts[-3]
    voltage = parts[-2]
    return temperature, voltage

def generate_pattern(plane: str, merged=False) -> None:
    """
    Generates the expected .img file name based on the plane input.
    Uses the exact function format you specified.
    """
    prefix = "CsV3Sb5_strain_merged_" if merged else "CsV3Sb5_strain_"
    if plane.startswith("(") and plane.endswith(")"):
        plane_content = plane.strip("()").replace(",", "_")
        return f"{prefix}{plane_content}.img"
    return None

def check_existing_data(local_data_path: str, Planes: list[str], merged: bool) -> bool:
    """
    Checks if all requested plane .img files exist in the local data folder.
    If they do, prompts the user to decide whether to replace them.

    Args:
        local_data_path (str): Path to the local data directory.
        Planes (list[str]): List of planes to check.
        merged (bool): Whether merged folder is used.

    Returns:
        bool: True if data gathering should proceed, False if skipped.
    """
    if not os.path.exists(local_data_path):
        return True  # Proceed if local data folder doesn't exist

    existing_files = set(os.listdir(local_data_path))
    expected_files = {generate_pattern(plane, merged) for plane in Planes}

    missing_files = expected_files - existing_files  # Files that are missing
    all_exist = len(missing_files) == 0

    if all_exist:
        user_input = input(f"All requested files are already present in {local_data_path}. Replace? (y/n): ").strip().lower()
        return user_input == "y"

    if missing_files:
        print(f"Missing files detected in {local_data_path}. Proceeding with data gathering.")

    return True  # Continue processing

def process_data(base_dir: str, local_dir: str, Planes: list[str], TEMPERATURES: list[str], VOLTAGES: dict[str, list[str]]) -> None:
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

    for temperature in TEMPERATURES:
        temp_path = os.path.join(data_dir, temperature)
        if not os.path.isdir(temp_path):
            continue

        plot_dic ={}
        for voltage in VOLTAGES.get(temperature, []):
            voltage_path = os.path.join(temp_path, voltage)
            local_data_path = os.path.join(voltage_path, "data")

            # Check if gathering is needed before accessing the remote base_dir
            gather_needed = check_existing_data(local_data_path, Planes, merged=True)
            if not gather_needed:
                print(f"Skipping data gathering for {temperature}, {voltage}.")
                # Still process existing files even if no gathering is needed
                img_files = [os.path.join(local_data_path, f) for f in os.listdir(local_data_path) 
                           if f.endswith(".img") and extract_plane_from_filename(f) in Planes]
                if img_files:
                    print('Processing existing files')
                    plot_dic[voltage] = process_img_files(img_files, voltage_path, temperature, voltage, Planes)
                continue

            # Now access the base_dir only if necessary
            remote_voltage_path = os.path.join(base_dir, temperature, voltage, "data")
            if not os.path.exists(remote_voltage_path):
                log_error(f"Remote data folder not found in base directory: {temperature}/{voltage}")
                continue

            # Process and copy required files only if missing or being replaced
            found_files = 0
            print(f"{voltage}")

            for file_name in os.listdir(remote_voltage_path):
                plane = extract_plane_from_filename(file_name)
                if plane and plane in Planes:
                    found_files += 1
                    src_file = os.path.join(remote_voltage_path, file_name)
                    dest_file = os.path.join(local_data_path, file_name)

                    os.makedirs(os.path.dirname(dest_file), exist_ok=True)
                    shutil.copy(src_file, dest_file)

                    # Handle `.Identifier` files
                    identifier_file = os.path.join(local_data_path, f"{os.path.splitext(file_name)[0]}.Identifier")
                    if os.path.exists(identifier_file):
                        os.remove(identifier_file)

            if found_files == 0:
                log_error(f"No matching files found for planes in {temperature}/{voltage} at {remote_voltage_path}")

            # Collect and process img files after copying
            img_files = [os.path.join(local_data_path, f) for f in os.listdir(local_data_path) 
                        if f.endswith(".img") and extract_plane_from_filename(f) in Planes]
            if img_files:
                print('Processing new and existing files')
                plot_dic[voltage] = process_img_files(img_files, voltage_path, temperature, voltage, Planes)
        plots_function(plot_dic)



# Inputs and code execution order

# Inputs: Define temperatures and voltages to process
TEMPERATURES = ["80K"]  # Add temperatures here

VOLTAGES = {
            "80K": ["0.0V", "8.0V", "16.0V", "24.0V", "38.0V"],
            }  # Voltages for each temperature

# Define the planes to be processed with regards to your inputed parameters in the processing functions
PLANES = ["(h,3,l)", "(3,k,l)"]

ratio = 0.01578947 #(l per pixel); Ratio to convert pixel units to l units calculated from gathered visual data where one concludes that 190 pixels correspond to 3l
N_pixel = 1476

# Code execution order
if __name__ == "__main__":
    # Base directory where the temperature folders are located, in the Cloud Storage
    base_dir = "/mnt/z/VEGA/CsV3Sb5_strain/2024/07/CsV3Sb5_July24_Kalpha/runs"
    local_dir = os.getcwd() 
    process_data(base_dir, local_dir, PLANES, TEMPERATURES, VOLTAGES)














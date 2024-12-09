## This code reads the .img files from CrysAlis, treats data in a certain way and plots it
## This then allows one to work with ROIs silx tool in order to extract valuable data regarding peaks intensities for a range of different hkl points

import fabio    # to read the .img files
from silx.gui.plot import Plot2D
import PyQt5.QtCore  # Importing PyQt5 will force silx to use it
from silx.gui import qt
from silx.gui.colors import Colormap
import os
import re
import shutil
import numpy as np
import matplotlib.pyplot as plt

# Visualize using silx
def visualize(img_file):
    # Initialize the Qt application
    app = qt.QApplication([])

    # Create a 2D plot
    plot = Plot2D()
    plot.setWindowTitle(f"CrystAlis {img_file} data")

    # Create a Colormap object; here minimum value is 0 and maximum 800 but for different data (sets) this might need adjustment
    # Merged data -> (min=0, max=800); Not merged data -> (min=0, max=100)
    # But adjust as you think it's more sutiable to you or your data
    if "merged" in img_file: 
        max_cap = 800
    else: 
        max_cap = 200
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

# Data processing and treatment unit

# Helper to extract temperature and voltage from folder names
def extract_temp_voltage(path):
    # Assumes structure: <base_dir>/<temperature>/<voltage>/data
    parts = path.split(os.sep)
    temperature = parts[-3]
    voltage = parts[-2]
    return temperature, voltage

# Function to extract the plane from the filename
def extract_plane_from_filename(filename):
    match = re.search(r'([\-\d\.]+)_k_l|h_k_([\-\d\.]+)|([\-\d\.]+)_k_([\-\d\.]+)|h_(\d+(\.\d+)?)_l', filename)
    if match:
        if match.group(1):  # Matches 0.5_k_l or -3_k_l
            return f"({match.group(1)},k,l)"
        elif match.group(2):  # Matches h_k_0.25
            return f"(h,k,{match.group(2)})"
        elif match.group(3) and match.group(4):  # Matches -3_k_0.25
            return f"({match.group(3)},k,{match.group(4)})"
        elif match.group(5):  # Matches h_3_l
            return f"(h,{match.group(5)},l)"
    print(f"Warning: Could not determine plane for filename: {filename}")
    return None

# Reorders the planes data gathering order with our input plane order so there are no miss placements while robustly handling missing planes during reordering
def reorder_data(img_files, planes):
    file_planes = []
    is_merged = []

    for f in img_files:
        plane = extract_plane_from_filename(os.path.basename(f))
        if plane is None:
            print(f"Warning: Could not determine plane for file {f}")
        file_planes.append(plane)
        is_merged.append("merged" in os.path.basename(f))

    if None in file_planes:
        log_error(f"Some files do not have recognizable plane information in the dataset.")
        return [], []  # Gracefully return empty lists to avoid halting

    ordered_data = []
    ordered_is_merged = []
    for plane in planes:
        try:
            index = file_planes.index(plane)  # Find the index of the plane in file_planes
            ordered_data.append(fabio.open(img_files[index]).data)
            ordered_is_merged.append(is_merged[index])
        except ValueError:
            continue  # Skip missing planes and move on

    return ordered_data, ordered_is_merged

# Main data processing function
def process_img_files(img_files, output_dir, temperature, voltage, Planes):
    # Reorder data to align with planes
    data, is_merged = reorder_data(img_files, Planes)
    
    #N_pixel = np.sqrt(np.size(data)/2)  # If one doen't know the pixel data size
    
    # Visualize the data previoustly if needed/wanted, at first this is all one needs to do in order to extract the information needed to run the code systematically
    """if temperature == "15K":
        plane_index = 0 # example
        visualize(data[plane_index])
        visualize(data[plane_index+1])"""

    # Run the main logic
    def initial_parameters(plane):
        # This is going to change from case to case, that's why first one does a visual analysis, this data (parameters) is taken from the ROI.dat file and inputed here
        params = {
            "(h,3,l)": (491, 948, 676, 5),
            "(h,0.5,l)": (591, 948, 1048, 5),  
            "(-3,k,l)": (400, 800, 1047, 5),
            "(3,k,l)": (700, 950, 676, 5),
            "(h,0,l)": (725, 1175, 1171, 5)

        }
        return params.get(plane, (0, 0, 0, 0))
    
    # Defines a starting and ending collumn for the exctraction of the peak intensity data
    def calculate_columns(width, column):
        start = column - width // 2
        end = column + width // 2 + 1
        return start, end
    
    # Filters out fake very intense peaks from the data if encontered
    def filter_large_values(data, multiplier=10):
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
    
    def find_peaks(data, height=None, min_prominence=20, min_distance=5):
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

    current_max, counter = 0, 0

    # Run the data processing for each plane
    for plane, plane_data, merged in zip(Planes, data, is_merged):
        par = initial_parameters(plane)
        roi_start, roi_end, central_column, column_width = par
        col_start, col_end = calculate_columns(column_width, central_column)
        roi_data = plane_data[roi_start:roi_end, col_start:col_end]
        row_means = np.mean(roi_data, axis=1)   # Calculate the mean across each row for the given the column values
        
        """
        # Change parameters values for merged data and not merged data if necessary
        if merged == True:
            cap_peak = 40
        else:
            cap_peak = 25"""
        
        # Get the peak's data and save them in the log.data file
        find_peaks_data = find_peaks(row_means)
        row_means_filtered = filter_large_values(row_means, multiplier=30)[0]
        size = np.size(row_means_filtered)
        x_0, x_1 = par[0] - N_pixel/2, par[1] - N_pixel/2

        log_data(f"Temperature: {temperature}, Voltage: {voltage}, Plane: {plane}", f"Peaks: {[round(float(p+x_0)*ratio,2) for p in find_peaks_data[0]]}, Average peak intensity: {round(find_peaks_data[2],2)}, Order: {round(find_peaks_data[1]/(size*ratio)+1,1)}")

        plt.plot(ratio * np.linspace(x_0, x_1, size), row_means_filtered, label=labels[counter], color=colors[counter])
        current_max = max(current_max, max(row_means_filtered))
        counter += 1

    plt.xticks(np.arange(-2, 4, 0.5), rotation=0)
    plt.ylim(0, current_max + 25)   # Add a value range (our case 20) so that the legend does not overlap with the data from the plots
    plt.xlim(-1.6, 3.1)
    plt.ylim(15)
    plt.xlabel("l (r.l.u)")
    plt.ylabel("Intensity")
    plt.legend(loc="upper left")
    plt.title(f"Intensity vs L\nTemperature: {temperature}, Voltage: {voltage}")

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"IvsL_{temperature}_{voltage}.png")
    plt.savefig(output_path, dpi=300)
    plt.close()

# Traverse folder structure to store the data for each point
def main_process(base_dir, Planes):
    data_dir = os.path.join(base_dir, "Data")  # Add 'Data' folder in the traversal
    for temp_dir in os.listdir(data_dir):
        temp_path = os.path.join(data_dir, temp_dir)
        if os.path.isdir(temp_path):
            for voltage_dir in os.listdir(temp_path):
                voltage_path = os.path.join(temp_path, voltage_dir)
                data_path = os.path.join(voltage_path, "data")
                
                if os.path.isdir(data_path):
                    img_files = [os.path.join(data_path, f) for f in os.listdir(data_path) if f.endswith(".img")]
                    
                    if img_files:
                        temperature, voltage = extract_temp_voltage(data_path)
                        process_img_files(img_files, voltage_path, temperature, voltage, Planes)

# Data gathering and transfering to local directory unit

# Log peak data to the log.data file
def log_data(message, info, log_file="log.data"):
    with open(log_file, "a") as f:  # Append to log file
        f.write(message + "\n")
        f.write(info + "\n")
        f.write("\n")

# Logging function for errors during code execution
def log_error(message, log_file="log.error"):
    print(message)  # Print to console
    with open(log_file, "a") as f:  # Append to log file
        f.write(message + "\n")
        f.write("\n")

# Function to process temperature and voltage folders in the Cloud Storage directory
def process_temperature_and_voltage(base_dir, local_dir, Planes):
    for temp in TEMPERATURES:
        temp_path = os.path.join(base_dir, temp)
        print(f"Processing {temp_path}")
        if not os.path.exists(temp_path):
            log_error(f"Temperature folder not found in Z: {temp}")
            continue

        for voltage in VOLTAGES.get(temp, []):
            voltage_path = os.path.join(temp_path, f"V{voltage}_-{voltage}")  # Voltage folder structure in Z
            if not os.path.exists(voltage_path):
                log_error(f"Voltage folder not found in Z: {temp}/{voltage}")
                continue

            fulltth_path = os.path.join(voltage_path, "fulltth")
            fulltth_ext_path = os.path.join(voltage_path, "fulltth_extended")

            # Determine the correct subfolder to process
            parent_path = fulltth_path if os.path.exists(fulltth_path) else fulltth_ext_path
            if not os.path.exists(parent_path):
                log_error(f"fulltth folder not found in Z: {temp}/{voltage}")
                continue

            merged_path = os.path.join(parent_path, "merged")
            esp_path = os.path.join(parent_path, "esp")
            unwarp_path = None

            # Check merged or esp folder
            if os.path.exists(merged_path) and os.listdir(merged_path):
                unwarp_path = os.path.join(merged_path, "unwarp")
                MERGED = True
            elif os.path.exists(esp_path) and os.listdir(esp_path):
                unwarp_path = os.path.join(esp_path, "unwarp")
                MERGED = False
            else:
                log_error(f"Neither 'merged' nor 'esp' folder is valid in Z: {temp}/{voltage}")
                continue

            if not os.path.exists(unwarp_path):
                log_error(f"'unwarp' folder not found in Z: {temp}/{voltage}")
                continue

            # Process files in unwarp folder
            process_unwarp_folder(unwarp_path, temp, voltage, local_dir, Planes, MERGED)

# Function to process files in 'unwarp' folder, where the .img data files are located
def process_unwarp_folder(unwarp_path, temp, voltage, local_dir, Planes, M):
    """
    Processes files in the unwarp folder to find matching planes.
    Logs temperature and voltage if a file is not found.
    """
    def generate_pattern(plane, merged=False):
        prefix = "CsV3Sb5_strain_merged_" if merged else "CsV3Sb5_strain_"
        if plane.startswith("(") and plane.endswith(")"):
            plane_content = plane.strip("()").replace(",", "_")
            return f"{prefix}{plane_content}.img"
        return None

    plane_patterns = [generate_pattern(plane, M) for plane in Planes]

    found_files = 0  # Track if any files are found for logging purposes
    print(f"{voltage}V")
    for pattern in plane_patterns:
        if pattern:
            print(f"Looking for files matching: {pattern}")  # Debug log
            file_found = False
            for file_name in os.listdir(unwarp_path):
                if pattern in file_name:
                    file_found = True
                    found_files += 1
                    src_file = os.path.join(unwarp_path, file_name)
                    dest_file = os.path.join(local_dir, "Data", temp, f"{voltage}V", "data", file_name)

                    # Copy the file
                    os.makedirs(os.path.dirname(dest_file), exist_ok=True)
                    shutil.copy(src_file, dest_file)

                    # Handle `.Identifier` files
                    identifier_file = os.path.join(local_dir, "Data", temp, f"{voltage}V", "data", f"{os.path.splitext(file_name)[0]}.Identifier")
                    if os.path.exists(identifier_file):
                        os.remove(identifier_file)

            if not file_found:
                log_error(f"Plane {pattern} not found for {temp}, {voltage}V at {unwarp_path}")

    if found_files == 0:
        log_error(f"No matching files found for planes in {temp}/{voltage} at {unwarp_path}")

# Main function to call the main function and process temperature and voltage folders in the Cloud Storage directory
def main(base_dir, local_dir, Planes):
    process_temperature_and_voltage(base_dir, local_dir, Planes)

# Inputs and code execution order

# Inputs: Define temperatures and voltages to process
TEMPERATURES = ["15K", "80K", "35K", "55K", "15K_low_strain", 
                 "15K_medium_strain", "15K_high_strain", "80K_low_strain", "80K_medium_strain", "80K_high_strain",
                   "100K_low_strain", "100K_medium_strain", "100K_high_strain"]  # Add temperatures here
VOLTAGES = {"15K": ["5.0", "20.0", "57.0", "125.0"], "80K": ["0.0", "8.0", "12.0", "38.0", "55.0"], "35K": ["86.0"], "55K": ["67.0"], 
            "15K_low_strain": ["20.0"], "15K_medium_strain": ["83.0"], "15K_high_strain": ["115.0"],
            "80K_low_strain": ["26.0"], "80K_medium_strain": ["30.0"], "80K_high_strain": ["95.0"],
            "100K_low_strain": ["15.0"], "100K_medium_strain": ["55.0"], "100K_high_strain": ["75.0"]}  # Voltages for each temperature

# Define the planes to be processed with regards to your inputed parameters in the processing functions
planes = ["(h,3,l)", "(h,0.5,l)", "(-3,k,l)", "(3,k,l)", "(h,0,l)"]
# Change this to fit your data/needs, might need to alter the code slightly if things are too different
labels, colors = ["(-0.5,3.0,l)", "(2.5,0.5,l)", "(-3,2.5,l)", "(3,-0.5,l)", "(3.5,0,l)"], ["red", "green", "blue", "blue", "red"]  # If they have the same color it means that they are equivelent points

ratio = 0.01578947 #(l per pixel); Ratio to convert pixel units to l units calculated from gathered visual data where one concludes that 190 pixels correspond to 3l
N_pixel = 1476

# Code execution order
if __name__ == "__main__":
    # Base directory where the temperature folders are located, in the Cloud Storage
    base_dir = "/mnt/z/VEGA/CsV3Sb5_strain/2024/07/CsV3Sb5_July24_Kalpha/runs"
    local_dir = os.getcwd() 
    # Routine checks to access what you need, in order to avoid unnecessary calculations, and for you to check if everything is set up correctly
    access = input("Do you have access to the Z: drive (is your VPN on)? (y/n): ")
    if access.lower() == "y":
        mounted = input("Is the Z: drive mounted in a folder named 'z' in your /mnt/ directory? (y/n): ")
        if mounted.lower() == "y":
            need_data_to_process = input("Do you need to copy data from the Z: drive? (y/n): ")
            if need_data_to_process.lower() == "y":
                print("Starting data gathering...")
                main(base_dir, local_dir, planes)
                print("Data gathering completed.")
                print("Starting data processing.")
                main_process(local_dir, planes)
                print("Data processing completed.")
            else:
                print("Starting data processing.")
                main_process(local_dir, planes)
                print("Data processing completed.")
        else:
            print("Exiting script due to Z: drive not being mounted in /mnt/z/.")
            print("Please make sure the Z: drive is mounted in a folder named 'z' in your /mnt/ directory and try again.")
            print("Care that every time you disconnect from the VPN you will need to mount the Z: drive again.")
            print("To mount the Z: drive, execute the following command in your terminal: sudo mount -t drvfs Z: /mnt/z")
            exit()
    else:
        print("Exiting script due to lack of access to the Z: drive.")
        print("Please make sure your VPN is on and try again.")
        exit()

## This code reads the .img files from CrysAlis, treats data in a certain way and plots it
## This then allows one to work with ROIs silx tool in order to extract valuable data regarding peaks intensities for a range of different hkl points

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

# Data processing and treatment unit

# Helper to extract temperature and voltage from folder names
def extract_temp_voltage(path: str) -> tuple[str, str]:
    # Assumes structure: <base_dir>/<temperature>/<voltage>/data
    parts = path.split(os.sep)
    temperature = parts[-3]
    voltage = parts[-2]
    return temperature, voltage

# Function to extract the plane from the filename
def extract_plane_from_filename(filename: str) -> tuple | None:
    match = re.search(r'([\-\d\.]+)_k_l|h_k_([\-\d\.]+)|([\-\d\.]+)_k_([\-\d\.]+)|h_(\d+(\.\d+)?)_l', filename)
    if match:
        if match.group(1):  # Matches 0.5_k_l or -3_k_l
            return f"({match.group(1)},k,l)", None
        elif match.group(2):  # Matches h_k_0.25
            return f"(h,k,{match.group(2)[:-1]})", float(match.group(2)[:-1])
        elif match.group(3) and match.group(4):  # Matches -3_k_0.25
            return f"({match.group(3)},k,{match.group(4)})", None
        elif match.group(5):  # Matches h_3_l
            return f"(h,{match.group(5)},l)", None
    print(f"Warning: Could not determine plane for filename: {filename}")
    return None

# Reorders the planes data gathering order with our input plane order so there are no miss placements while robustly handling missing planes during reordering
def reorder_data(img_files: list[str], planes: list[str]) -> tuple[list[np.ndarray], list[bool], list[float]]:
    file_planes = []
    is_merged = []
    l_list = []

    for f in img_files:
        plane, l = extract_plane_from_filename(os.path.basename(f))
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

# Main data processing function
def process_img_files(img_files: list[str], output_dir: str, temperature: str, voltage: str, Planes: list[str]) -> None:
    # Reorder data to align with planes
    data, is_merged, l_values = reorder_data(img_files, Planes)
    """in_plane = False
    if len(l_values) > 0:
        in_plane = True 
    else:
        log_error(f"No in-plane Planes for data analysis for: ({temperature}, {voltage})")"""
    
    #N_pixel = np.sqrt(np.size(data)/2)  # If one doen't know the pixel data size

    # Visualize the data previoustly if needed/wanted, at first this is all one needs to do in order to extract the information needed to run the code systematically
    """if temperature == "80K":
        plane_index = 0 # example
        visualize(data[plane_index])
        visualize(data[plane_index+1])"""

    # Run the main logic
    if in_plane == False:
        log_data("Running for Intensity vs l (values) plots.", "")

        def l_plots_run():

            def initial_parameters(plane: str) -> tuple[int, int, int, int]:
                # This is going to change from case to case, that's why first one does a visual analysis, this data (parameters) is taken from the ROI.dat file and inputed here
                params = {
                    "(h,3,l)": (491, 948, 676, 5),
                    #"(h,0.5,l)": (591, 948, 1048, 5),  
                    #"(-3,k,l)": (400, 800, 1047, 5),
                    "(3,k,l)": (700, 950, 676, 5),
                    "(h,0,l)": (725, 1175, 1171, 5)

                }
                return params.get(plane, (0, 0, 0, 0))
            
            # Defines a starting and ending collumn for the exctraction of the peak intensity data
            def calculate_columns(width: int, column: int) -> tuple[int, int]:
                start = column - width // 2
                end = column + width // 2 + 1
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

            current_max, counter = 0, 0

            # Run the data processing for each plane
            for plane, plane_data in zip(Planes, data):
                par = initial_parameters(plane)
                roi_start, roi_end, central_column, column_width = par
                col_start, col_end = calculate_columns(column_width, central_column)
                roi_data = plane_data[roi_start:roi_end, col_start:col_end]
                row_means = np.mean(roi_data, axis=1)   # Calculate the mean across each row for the given the column values
                
                # Get the peak's data and save them in the log.data file
                find_peaks_data = find_peaks(row_means)
                row_means_filtered = filter_large_values(row_means, multiplier=30)[0]
                row_means_filtered[row_means_filtered < 10] = 0.0
                size = np.size(row_means_filtered)
                x_0, x_1 = par[0] - N_pixel/2, par[1] - N_pixel/2

                log_data(f"Temperature: {temperature}, Voltage: {voltage}, Plane: {plane}", f"Peaks: {[round(float(p+x_0)*ratio,2) for p in find_peaks_data[0]]}, Average peak intensity: {round(find_peaks_data[2],2)}, Order: {round(find_peaks_data[1]/(size*ratio)+1,1)}")

                plt.plot(ratio * np.linspace(x_0, x_1, size), row_means_filtered, label=labels[counter], color=colors[counter])
                current_max = max(current_max, max(row_means_filtered))
                counter += 1

            """if is_merged[0] == False:
                noise_cap = 15
            else:
                noise_cap = 30
"""
            plt.xticks(np.arange(-2, 4, 0.5), rotation=0)
            plt.ylim(0, current_max + 25)   # Add a value range (our case 20) so that the legend does not overlap with the data from the plots
            plt.xlim(-0.7, 3.1)
            #plt.ylim(noise_cap)
            plt.xlabel("l (r.l.u)")
            plt.ylabel("Intensity")
            plt.legend(loc="upper left")
            plt.title(f"Intensity vs L\nTemperature: {temperature}, Voltage: {voltage}")

            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, f"IvsL_{temperature}_{voltage}.png")
            plt.savefig(output_path, dpi=300)
            plt.close()

        l_plots_run()
        return print("Runned for Intensity vs l (values) plots.")
    
    else:
        log_data("Running for Intensity values calculation for in-plane cuts.", "")

        def intensity_inPlane_run():

            def detect_chirality(peak_intensities: list[float]) -> bool:
                # Ensure the list has exactly 6 peaks
                if len(peak_intensities) != 6:
                    raise ValueError("The input list must have exactly 6 intensity values.")

                # Calculate average intensities of symmetric pairs
                avg_03 = (peak_intensities[0] + peak_intensities[3]) / 2
                avg_14 = (peak_intensities[1] + peak_intensities[4]) / 2
                avg_25 = (peak_intensities[2] + peak_intensities[5]) / 2

                # Use avg_14 as the reference (pivot)
                if avg_03 < avg_14 < avg_25:
                    return True
                elif avg_03 > avg_14 > avg_25:
                    return False
                else:
                    return None

            def plot_HKplane(plot_range: tuple, intensities: list[float], PeaksPos: dict[int, tuple], figure_size: tuple, pixel_data: np.array, title_text: str, temperature: str, voltage: str, M: bool) -> None:
                """
                Function to plot HK-plane data with peaks annotated, using Matplotlib.
                
                Args:
                    plot_range (tuple): (y_min, y_max, x_min, x_max) range to subset pixel data.
                    intensities (list): List of intensities corresponding to peak positions.
                    PeaksPos (dict): Dictionary mapping indices to peak positions as tuples (x, y).
                    figure_size (tuple): Size of the plot figure (width, height).
                    pixel_data (np.array): 2D array of pixel intensity values.
                    title_text (str): Title of the plot.
                    temperature (str): Temperature annotation for plot title.
                    voltage (str): Voltage annotation for plot title.
                """
                
                # Subset the pixel data based on the specified plot range
                subset_data = pixel_data[plot_range[0]:plot_range[1], plot_range[2]:plot_range[3]]
                
                # Create the figure and axis with the specified size
                fig, ax = plt.subplots(figsize=figure_size)
                ax.set_title(f"({temperature}, {voltage}); {title_text}")
                
                if M == True:
                    max_cap = 100
                else:
                    max_cap = 50

                norm = Normalize(vmin=0, vmax=max_cap)
                
                # Plot the subset data with a colormap
                img = ax.imshow(subset_data, cmap="viridis", norm=norm, origin='lower', 
                                extent=(plot_range[2], plot_range[3], plot_range[0], plot_range[1]))
                #img = ax.imshow(subset_data, cmap="viridis", origin='lower', 
                #                extent=(plot_range[2], plot_range[3], plot_range[0], plot_range[1]))
                plt.colorbar(img, ax=ax, label='Intensity')
                
                # Adjust peaks' positions relative to the plot_range
                delta_x, delta_y = plot_range[2], plot_range[0]
                new_peaks_positions = {index: (x - delta_x, y - delta_y) for index, (x, y) in PeaksPos.items()}

                # Add text and red circle markers for each peak
                for index, pos in new_peaks_positions.items():
                    x, y = pos
                    ax.text(x + delta_x, y + delta_y + figure_size[0], f'{round(intensities[index], 1)}', color='red', ha='center', va='bottom', fontsize=9)
                    # Add a red circle around each peak
                    circle = Circle((x + delta_x, y + delta_y), radius=figure_size[0], edgecolor='red', facecolor='none', lw=1.0)
                    ax.add_patch(circle)
                
                # Set labels and show the plot
                ax.set_xlabel("Pixel X")
                ax.set_ylabel("Pixel Y")
                plt.tight_layout()
                to_go_back = os.getcwd()
                os.chdir(f"{to_go_back}/Data/{temperature}/{voltage}/")
                # Save the figure with specified dpi (higher dpi = higher resolution)
                plt.savefig(fname = f"{plane}.png", dpi=300, bbox_inches='tight')
                plt.close()
                os.chdir(to_go_back)
                #plt.show()

            UnitCell_parameters = (990, 1130, 840, 1000)   # (row_start, row_end, col_start, col_end); defines the boxe delimitation for the hk space of comparison
            peak_BoxSize = (7, 7)
            peaks_positions = {0: (891,1005), 1: (860, 1058), 2: (891, 1112),
                               3: (952,1112), 4: (984, 1059), 5: (953, 1005)} # as (collum, row)

            PeaksPositions_RangeForCalc = {}
            index = 0
            for peak_pos in peaks_positions.values():
                x, y = peak_pos[0], peak_pos[1]
                delta_x, delta_y = peak_BoxSize[0], peak_BoxSize[1]
                PeaksPositions_RangeForCalc[index] = (y - delta_y, y + delta_y, x - delta_x, x + delta_x)
                index += 1

            chirality_list = [] # It assumes -> True: Right; False: Left; None: Non detected
            for plane, plane_data, merged in zip(Planes, data, is_merged):
                index = 0
                peaks_intensities = {}
                for inputs in PeaksPositions_RangeForCalc.values():
                    peak_data = plane_data[inputs[0]: inputs[1], inputs[2]: inputs[3]]
                    data_intensity = np.mean(peak_data)
                    peaks_intensities[index] = round(float(data_intensity),2)
                    index += 1
                
                background_I = round(float(np.mean(plane_data[990:1000, 840:850])),2)
                if merged == True:
                    merged_I_avg.append(background_I)
                else:
                    non_merged_I_avg.append(background_I)

                log_data(f"Temperature = {temperature} and Voltage = {voltage}", "")
                log_data(f"Plane {plane} (is merged = {merged}; Background intensity = {background_I}):", f"Peaks intensities: {peaks_intensities}")

                peaks_intensities_list = list(peaks_intensities.values())
                chirality = detect_chirality(peaks_intensities_list)
                chirality_list.append(chirality)

                title = str(plane) + "; Chirality: " + str(chirality)
                plot_HKplane(UnitCell_parameters, peaks_intensities_list, peaks_positions, peak_BoxSize, plane_data, title, temperature, voltage, merged)


        intensity_inPlane_run()
        return print("Runned for Intensity values calculation for in-plane cuts.")

# Traverse folder structure to store the data for each point
def main_processing(base_dir: str, Planes: list[str]) -> None:
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

# Function to process temperature and voltage folders in the Cloud Storage directory
def process_temperature_and_voltage(base_dir: str, local_dir: str, Planes: list[str]) -> None:
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
def process_unwarp_folder(unwarp_path: str, temp: str, voltage: str, local_dir: str, Planes: list[str], M: bool) -> None:
    """
    Processes files in the unwarp folder to find matching planes.
    Logs temperature and voltage if a file is not found.
    """
    def generate_pattern(plane: str, merged=False) -> None:
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
def main_gathering(base_dir: str, local_dir: str, Planes: list[str]) -> None:
    process_temperature_and_voltage(base_dir, local_dir, Planes)

# Inputs and code execution order

# Inputs: Define temperatures and voltages to process
TEMPERATURES = ["15K", "80K", "45K", "75K", "15K_low_strain", 
                 "15K_medium_strain", "15K_high_strain", "80K_low_strain", "80K_medium_strain", "80K_high_strain"]  # Add temperatures here

VOLTAGES = {"15K": ["5.0", "20.0", "57.0", "125.0"], 
            "80K": ["0.0", "8.0", "12.0", "14.5", "16.0", "18.0", "20.0", "24.0", "27.0", "29.0", "32.0", "38.0", "41.0", "42.0", "47.0", "48.0", "55.0"],
            "45K": ["74.0"], 
            "75K": ["58.0"], 
            "15K_low_strain": ["20.0"], "15K_medium_strain": ["83.0"], "15K_high_strain": ["115.0"],
            "80K_low_strain": ["26.0"], "80K_medium_strain": ["30.0"], "80K_high_strain": ["95.0"]}  # Voltages for each temperature

# Define the planes to be processed with regards to your inputed parameters in the processing functions
planes = ["(h,k,-0.25)", "(h,k,-0.5)", "(h,k,0)"]
#planes = ["(h,3,l)", "(3,k,l)", "(h,0,l)"]
# Change this to fit your data/needs, might need to alter the code slightly if things are too different
labels, colors = ["(-0.5,3.0,l)", "(2.5,0.5,l)", "(-3,2.5,l)", "(3,-0.5,l)", "(3.5,0,l)"], ["red", "blue", "red", "blue", "red"]  # If they have the same color it means that they are equivelent points

ratio = 0.01578947 #(l per pixel); Ratio to convert pixel units to l units calculated from gathered visual data where one concludes that 190 pixels correspond to 3l
N_pixel = 1476

in_plane = True # choose what to run the code for

# For benchmarking the difference between merged and non-merged data

merged_I_avg = []
non_merged_I_avg = []

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
                main_gathering(base_dir, local_dir, planes)
                print("Data gathering completed.")
                print("Starting data processing.")
                main_processing(local_dir, planes)
                print("Data processing completed.")

                if in_plane == True:
                    print("Benchmarking results")

                    ratio = (sum(merged_I_avg)/len(merged_I_avg))/(sum(non_merged_I_avg)/len(non_merged_I_avg))

                    print(f"Ratio between merged and non-merged intensity data is on average: {round(ratio, 2)}")
                    log_data("\n", f"Ratio between merged and non-merged intensity data is on average: {round(ratio, 2)}")

            else:
                print("Starting data processing.")
                main_processing(local_dir, planes)
                print("Data processing completed.")

                if in_plane == True:
                    print("Benchmarking results")

                    ratio = (sum(merged_I_avg)/len(merged_I_avg))/(sum(non_merged_I_avg)/len(non_merged_I_avg))

                    print(f"Ratio between merged and non-merged intensity data is on average: {round(ratio, 2)}")
                    log_data("\n", f"Ratio between merged and non-merged intensity data is on average: {round(ratio, 2)}")

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

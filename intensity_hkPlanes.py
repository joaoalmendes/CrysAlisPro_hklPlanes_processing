import fabio    # to read the .img files
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np
from matplotlib.colors import Normalize
from matplotlib.colorbar import ColorbarBase
from matplotlib.ticker import FuncFormatter

# It assumes -> True: Right; False: Left; None: Non detected
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

def plot_hk_plane(plot_range: tuple, intensities: list[float], PeaksPos: dict[int, tuple], circle_size: tuple, pixel_data: np.array, title_text: str, temperature: str, voltage: str, Merged: bool, Plane: str, local_Dir: str, ratio: float, N_pixel: int) -> None:
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
    subset_data = pixel_data[plot_range[2]:plot_range[3], plot_range[0]:plot_range[1]]
    
    # Create the figure and axis with the specified size
    fig, ax = plt.subplots(figsize=(7,7))
    ax.set_title(f"({temperature}, {voltage}); {title_text}")
    
    if Merged == True:
        max_cap = 100
    else:
        max_cap = 50

    norm = Normalize(vmin=0, vmax=max_cap)
    
    # Plot the subset data with a colormap
    img = ax.imshow(subset_data, cmap="viridis", norm=norm, origin='lower', 
                    extent=(plot_range[0], plot_range[1], plot_range[2], plot_range[3]))
    plt.colorbar(img, ax=ax, label='Intensity')
    
    # Adjust peaks' positions relative to the plot_range
    delta_x, delta_y = plot_range[0], plot_range[2]
    rescaled_positions = {peak_index: (x - delta_x, y - delta_y) for peak_index, (x, y) in PeaksPos.items()}

    # Add text and red circle markers for each peak
    for peak_index, pos in rescaled_positions.items():
        x, y = pos
        ax.text(x + delta_x, y + delta_y + circle_size[0], f'{round(intensities[peak_index], 1)}', color='red', ha='center', va='bottom', fontsize=9)
        # Add a red circle around each peak
        circle = Circle((x + delta_x, y + delta_y), radius=circle_size[0], edgecolor='red', facecolor='none', lw=1.0)
        ax.add_patch(circle)
    

    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, pos: f"{(value - N_pixel/2) * ratio -1.5 :.2f}"))    ## N_pixel is not the same for these cuts, need to check on CrysAlis the right value
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, pos: f"{(value - N_pixel/2) * ratio / np.sin(np.radians(60)):.2f}"))

    # Set labels and show the plot
    ax.set_xlabel("H")
    ax.set_ylabel("K")
    plt.tight_layout()
    os.chdir(f"{local_Dir}/Data/{temperature}/{voltage}/")
    # Save the figure with specified dpi (higher dpi = higher resolution)
    plt.savefig(fname = f"{Plane}.png", dpi=300, bbox_inches='tight')
    plt.close()
    os.chdir(local_Dir)
    #plt.show()

def intensity_hk_plane(data, Planes, is_merged, temperature, voltage, local_dir, ratio, N_pixel):
    # For benchmarking the difference between merged and non-merged data

    merged_I_avg = []
    non_merged_I_avg = []

    hk_parameters = (840, 1000, 990, 1130)   # (col_start, col_end, row_start, row_end)/(X_start, X_end, Y_start, Y_end); defines the boxe delimitation for the hk space of comparison

    peak_BoxSize = (7, 7)
    peaks_positions = {0: (891,1005), 1: (860, 1058), 2: (891, 1112),
                       3: (952,1112), 4: (984, 1059), 5: (953, 1005)} # as (collum, row)/(X, Y)


    PeaksPositions_RangeForCalc = {}
    index = 0
    for peak_pos in peaks_positions.values():
        x, y = peak_pos[0], peak_pos[1]
        delta_x, delta_y = peak_BoxSize[0], peak_BoxSize[1]
        PeaksPositions_RangeForCalc[index] = (y - delta_y, y + delta_y, x - delta_x, x + delta_x)
        index += 1

    chirality_list = []
    for plane, plane_data in zip(Planes, data):
        index = 0
        peaks_intensities = {}
        for inputs in PeaksPositions_RangeForCalc.values():
            peak_data = plane_data[inputs[0]: inputs[1], inputs[2]: inputs[3]]
            data_intensity = np.mean(peak_data)
            peaks_intensities[index] = round(float(data_intensity),2)
            index += 1

        peaks_intensities_list = list(peaks_intensities.values())
        chirality = detect_chirality(peaks_intensities_list)
        chirality_list.append(chirality)

        title = "Plane: " + str(plane) + "; Chirality: " + str(chirality)
        plot_hk_plane(hk_parameters, peaks_intensities_list, peaks_positions, peak_BoxSize, plane_data, title, temperature, voltage, is_merged, plane, local_dir, ratio, N_pixel)

def check_merged(img_file):
    if "merged" in img_file:
        return True
    else:
        return False







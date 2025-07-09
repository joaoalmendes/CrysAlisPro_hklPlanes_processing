import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.ticker import MultipleLocator, FuncFormatter
from matplotlib.patches import Circle
import os


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

def plot_hk_plane(plot_range: tuple, intensities: list[float], PeaksPos: dict[int, tuple], circle_size: tuple, pixel_data: np.array, title_text: str, temperature: str, voltage: str, Merged: bool, Plane: str, local_Dir: str, ratio: float, N_pixel: int, dataframe: list) -> None:
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
        max_cap = 500
        min_cap = 20
    else:
        max_cap = 250
        min_cap = 10

    norm = Normalize(vmin=min_cap, vmax=max_cap)
    
    dataframe.append([temperature, float(voltage[:-1]), Plane, subset_data.tolist()])   # Hysteric
    #dataframe.append([float(temperature[:-1]), float(voltage[:-1]), Plane, subset_data.tolist()])   # Normal

    # Plot the subset data with a colormap
    img = ax.imshow(subset_data, cmap="terrain_r", norm=norm, origin='lower', 
                    extent=(plot_range[0], plot_range[1], plot_range[2], plot_range[3]))
    plt.colorbar(img, ax=ax, label='Intensity')
    
    # Adjust peaks' positions relative to the plot_range
    delta_x, delta_y = plot_range[0], plot_range[2]
    rescaled_positions = {peak_index: (x - delta_x, y - delta_y) for peak_index, (x, y) in PeaksPos.items()}

    # Add text and red circle markers for each peak
    #for peak_index, pos in rescaled_positions.items():
        #x, y = pos
        #ax.text(x + delta_x, y + delta_y + circle_size[0], f'{round(intensities[peak_index], 1)}', color='red', ha='center', va='bottom', fontsize=9)
        # Add a red circle around each peak
        #circle = Circle((x + delta_x, y + delta_y), radius=circle_size[0], edgecolor='red', facecolor='none', lw=1.0)
        #ax.add_patch(circle)

    # Set spacing in HK space
    sin60 = np.sin(np.radians(60))
    tick_step = 0.25

    # Convert HK spacing to pixel units
    pixel_step_H = tick_step / ratio
    pixel_step_K = tick_step * sin60 / ratio

    # Apply to X axis
    ax.xaxis.set_major_locator(MultipleLocator(pixel_step_H))
    ax.xaxis.set_major_formatter(FuncFormatter(
        lambda value, pos: f"{(value - N_pixel / 2) * ratio - 1.5:.2f}"
    ))

    # Center of the y-axis in pixel coordinates
    subset_center_y = (plot_range[2] + plot_range[3]) / 2

    # Generate ticks within bounds
    tick_positions = []
    tick_labels = []

    i = 0
    K_center = 3
    while True:
        # Check both directions from the center
        pos_plus = subset_center_y + i * pixel_step_K
        pos_minus = subset_center_y - i * pixel_step_K
        val_plus = K_center + i * tick_step
        val_minus = K_center - i * tick_step

        added = False
        if plot_range[2] <= pos_plus <= plot_range[3]:
            tick_positions.append(pos_plus)
            tick_labels.append(f"{val_plus:.2f}")
            added = True

        if i != 0 and plot_range[2] <= pos_minus <= plot_range[3]:
            tick_positions.append(pos_minus)
            tick_labels.append(f"{val_minus:.2f}")
            added = True

        if not added:
            break
        i += 1

    # Sort ticks so they appear from bottom to top correctly
    tick_positions, tick_labels = zip(*sorted(zip(tick_positions, tick_labels)))

    # Apply to y-axis
    ax.set_yticks(tick_positions)
    ax.set_yticklabels(tick_labels)

    # Set labels and show the plot
    ax.set_xlabel("H")
    ax.set_ylabel("K (values)")
    #plt.tight_layout()
    os.chdir(f"{local_Dir}/plots/Region1__0_3_l/")
    # Save the figure with specified dpi (higher dpi = higher resolution)
    #plt.savefig(fname = f"{temperature}_{voltage}_{Plane}.png", dpi=300, bbox_inches='tight')
    #plt.show()
    #plt.close()
    os.chdir(local_Dir)

def intensity_hk_plane(data, Planes, is_merged, temperature, voltage, local_dir, ratio, N_pixel, dataframe):
    # For benchmarking the difference between merged and non-merged data

    merged_I_avg = []
    non_merged_I_avg = []

    # Region 1
    hk_parameters = (840, 1000, 990, 1130)   # (col_start, col_end, row_start, row_end)/(X_start, X_end, Y_start, Y_end); defines the boxe delimitation for the hk space of comparison
    #hk_parameters = (470, 630, 990, 1130)   # Region 2

    peak_BoxSize = (7, 7)
    # Region 1
    peaks_positions = {0: (891,1005), 1: (860, 1058), 2: (891, 1112),
                       3: (952,1112), 4: (984, 1059), 5: (953, 1005)} # as (collum, row)/(X, Y)
    # Region 2
    #peaks_positions = {0: (522,1005), 1: (491, 1058), 2: (523, 1112),
    #                   3: (583,1112), 4: (615, 1059), 5: (584, 1005)} # as (collum, row)/(X, Y)

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
        plot_hk_plane(hk_parameters, peaks_intensities_list, peaks_positions, peak_BoxSize, plane_data, title, temperature, voltage, is_merged, plane, local_dir, ratio, N_pixel, dataframe)

def check_merged(img_file):
    if "merged" in img_file:
        return True
    else:
        return False







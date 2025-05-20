import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.ticker import MultipleLocator, FuncFormatter
from matplotlib.patches import Circle
import os
from read_reciprocal_space_cuts import check_existing_data, extract_plane_from_filename, reorder_files_to_data, check_merged#, generate_pattern

def data_processing(data, Planes_list, is_merged, temperature, voltage, local_path, ratio, N_pixel):
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
    list_peaks_intensities = []
    for peaks_positions in list_peaks_positions:
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
            list_peaks_intensities.append(peaks_intensities)


    return None

def get_data(original_path, Planes_list, temperature, ratio, N_pixel):
    os.chdir(original_path)
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
                data_processing(data, Planes_list, is_merged, temperature, voltage, local_path, ratio, N_pixel)
        os.chdir(original_path)


T = "80K"
main_path = f"./Data/{T}/"
Planes = ["(h,k,0)", "(h,k,-0.25)", "(h,k,-0.5)"]
ratio_Pixels_to_iAngs, Number_pixels = 0.00813008, N_pixel = 1476

get_data(main_path, Planes, T, ratio_Pixels_to_iAngs, Number_pixels)





















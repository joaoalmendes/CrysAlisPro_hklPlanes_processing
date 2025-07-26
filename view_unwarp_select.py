import fabio    # to read the .img files
from silx.gui.plot import Plot2D
import PyQt5.QtCore  # Importing PyQt5 will force silx to use it
from silx.gui import qt
from silx.gui.colors import Colormap
from PyQt5.QtCore import QRectF
from PyQt5.QtGui import QPen, QColor
from PyQt5.QtWidgets import QGraphicsEllipseItem
import os
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from matplotlib.colorbar import ColorbarBase

# Visualize using silx
def visualize(path_to_file: np.array) -> None:
    img_file = fabio.open(path_to_file).data
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

def get_RSM_cut(file_path, ROI_coordinates, Merged = False):

    if "merged" in file_path:
        Merged = True

    if Merged == True:
        max_cap = 500
        min_cap = 20
    else:
        max_cap = 250
        min_cap = 10

    data = fabio.open(file_path).data

    plot_data = data[ROI_coordinates[2]:ROI_coordinates[3], ROI_coordinates[0]:ROI_coordinates[1]]
    
    # Create the figure and axis with the specified size
    fig, ax = plt.subplots(figsize=(7,7))
    norm = Normalize(vmin=min_cap, vmax=max_cap)
    img = ax.imshow(plot_data, cmap="terrain_r", norm=norm, origin='lower')
    plt.colorbar(img, ax=ax, label='Intensity')
    plt.show()
    plt.close()
    return None

local_dir = os.getcwd() 
Temperature = "80K"
Voltage = "38.0V"
plane = "h_3_l"

path_merged = f"{local_dir}/Data/{Temperature}/{Voltage}/data/CsV3Sb5_strain_merged_{plane}.img"
path_esp = f"{local_dir}/Data/{Temperature}/{Voltage}/data/CsV3Sb5_strain_{plane}.img"

delta_x, delta_y = 162, 100
ROI_x, ROI_y = 595, 590
ROI = (ROI_x, ROI_x + delta_x, ROI_y, ROI_y + delta_y)   # (col_start, col_end, row_start, row_end)/(X_start, X_end, Y_start, Y_end)

#visualize(path_merged)

get_RSM_cut(path_merged, ROI)

"""
--- Main Peaks

delta_x = 154; delta_y = 200

(h3l): X, Y = 477, 571

(3kl): X, Y = 599, 702

(1-h,2+k,l): X, Y = 475, 704

--- New Peaks

delta_x, delta_y = 162, 100

(0kl): X, Y = 968, 658

(h0l): X, Y = 970, 719

(h, 1-k, l): X, Y = 853, 787

(h, 3, l): X, Y = 595, 590

"""



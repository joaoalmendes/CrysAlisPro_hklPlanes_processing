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











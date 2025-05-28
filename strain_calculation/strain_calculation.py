from huberutils.utils import my_qconv
from huberutils.cbfTools import HuberCBFRun
import pyFAI
import fabio
import matplotlib.pyplot as plt
import numpy as np
import glob
import os
from scipy.ndimage import center_of_mass
import ase
import pandas as pd

plt.close('all')
from lmfit.models import LorentzianModel, ConstantModel, PseudoVoigtModel, QuadraticModel
from lmfit.models import LinearModel

def my_fit3(x,y, start=None):
    Kbwls = [
        0.713607,
    ]
    Kb_main = 0.70931715    # Wavelength 
    pv = PseudoVoigtModel()
    lm = LinearModel()
    model = pv + lm
    for i, wl in enumerate(Kbwls):
        model = model + PseudoVoigtModel(prefix=f"p{i}_")
    params = model.make_params()
    params.update(lm.guess(y,x))
    params.add("d", expr=f"{Kb_main}/(2*sin(center/2*pi/180))")
    params.add("q", expr=f"4*pi/{Kb_main}*sin(center/2*pi/180)")
    params.add("I12", value=0.42, vary=False)
    for i, wl in enumerate(Kbwls):
        params[f"p{i}_center"].set(expr=f"2*(asin({wl}/2/d))*180/pi")
        params[f"p{i}_sigma"].set(expr="sigma")
        params[f"p{i}_fraction"].set(expr="fraction")
        params[f"p{i}_amplitude"].set(expr="I12*amplitude")

    params.update(PseudoVoigtModel().guess(y - lm.eval(x=x, params=lm.guess(y,x)), x))
        #params.add("strech")
    params['p0_sigma'].set(expr="sigma")
    #params['p2_sigma'].set(expr="sigma*strech")
    #display(params)
    if start:
        params = start
    res = model.fit(y, params=params, x=x)
    return res

def get_voltage(name):
    # From the voltage string file name's, get's the float value of the voltage
    return float(name.split('/')[0].split('_')[0][1:])

def update_xy(dat, xy, dx, dy, _cnt=0):
    if _cnt > 10:
        print('maximum recursion reached')
        return xy
    xy = np.asarray(xy)
    edge = xy - np.array([dx,dy])
    xy_old = 1.*xy
    new_xy = center_of_mass(dat[xy[0]-dx:xy[0]+dx+1,xy[1]-dy:xy[1]+dy+1]) + edge
    print(new_xy)
    if np.linalg.norm(new_xy - xy) < 0.5:
        return np.array([int(_) for _ in np.round(new_xy)])
    update_xy(dat, new_xy, dx, dy, _cnt+1)

def get_data_and_plot(xy, dx, dy, npts, margin, poni, VEGA_dir, Temperature, home_dir, file_name, do_update_xy=False):
    f, axs = plt.subplots(1, 2, figsize=(10, 5))  # Create a figure with two subplots

    os.chdir(os.path.join(VEGA_dir, Temperature))

    head = glob.glob('V[0-9]*')
    head.sort(key=get_voltage)
    poni_file_info = pyFAI.load(poni)  # Load poni file for experimental parameters
    if Temperature == "80K":
        tail = '6peaks/merged/CsV3Sb5_strain_merged_01_'  # Frame where peak is detected
    if Temperature == "15K":
        tail = '7peaks/merged/CsV3Sb5_strain_02_'

    dataframe = []
    for i, pre in enumerate(head):
        volt = get_voltage(pre)
        print(os.path.join(pre, tail))
        try:
            run = HuberCBFRun(os.path.join(pre, tail))  # Calculator object for data frames
        except:
            continue
        
        max_I_frame = np.argmax(run.data[:, xy[0]-dx:xy[0]+dx, xy[1]-dy:xy[1]+dy].sum(axis=(1,2)))  # Peak intensity frame
        
        if do_update_xy:
            xy = update_xy(run.data[max_I_frame-margin:max_I_frame+margin+1].sum(axis=0), xy, dx, dy)

        poni_file_info.rot2 = run.th2 * np.pi / 180
        mask = np.ones(run.frames[0].data.shape)
        mask[xy[0]-dx:xy[0]+dx+1, xy[1]-dy:xy[1]+dy+1] = 0

        tth, I = poni_file_info.integrate1d(run.data[max_I_frame-margin:max_I_frame+margin+1].sum(axis=0), npt=npts, unit='2th_deg', mask=mask)
        th = (run.th_s + 0.5 * run.th_i)[max_I_frame-margin:max_I_frame+margin+1]
        Ith = run.data[max_I_frame-margin:max_I_frame+margin+1, xy[0]-dx:xy[0]+dx+1, xy[1]-dy:xy[1]+dy+1].sum(axis=(1,2))
        
        res = my_fit3(tth, I)  # Fit the 2theta vs Intensity data
        _l, = axs[0].plot(tth, I, 'o')
        axs[0].plot(tth, res.best_fit, color=_l.get_color(), ls='-', label=str(volt))
        axs[0].legend() 
        axs[0].set_xlabel('2Theta')
        axs[1].set_xlabel('Theta')
        axs[1].plot(th, Ith, color=_l.get_color(), marker='o')
        
        try:
            d_value = res.params['d'].value
            e_d_value = res.params['d'].stderr
            print(i, max_I_frame, xy, volt, d_value, res.params['p0_center'].value)
            # Append main data plus the arrays as lists
            dataframe.append([volt, d_value, e_d_value, list(tth), list(I), list(th), list(Ith)])
        except:
            print('fit failed')

    # Add y-axis labels after the loop
    axs[0].set_ylabel('Intensity')
    axs[1].set_ylabel('Ith')

    plt.show()
    plt.close()

    os.chdir(home_dir)

    # Create DataFrame with columns for main data and arrays
    df = pd.DataFrame(dataframe, columns=['Voltage (V)', 'd (A)', 'd_error (A)', 'tth (degrees)', 'I (Intensity)', 'th (degrees)', 'Ith (Peak intensity (theta))'])
    # Save to a single tab-separated CSV file
    df.to_csv(file_name, sep='\t', index=False)
    return None

def process_data(input_file, peak, Temperature):
    # Read the input file with tab-separated values
    df = pd.read_csv(input_file, sep='\t')
    
    # Sort by voltage to ensure we get the smallest voltage first
    df = df.sort_values(by='Voltage (V)')
    
    # Get reference values from the smallest voltage (first row after sorting)
    d_reference = df.iloc[0]['d (A)']
    error_d_reference = df.iloc[0]['d_error (A)']
    
    # Calculate strain (%) using the formula: (|d - d_reference|/d_reference) * 100
    df['strain (%)'] = (np.abs(df['d (A)'] - d_reference) / d_reference) * 100
    
    # Calculate error in strain (%) using the provided propagation formula
    df['error_strain (%)'] = 100 * np.sqrt(
        df['d_error (A)']**2 + ((2 / d_reference) * error_d_reference)**2
    )
    
    # Add the d_reference value as a column
    df['d_reference (A)'] = d_reference
    
    # Ensure the DataFrame has all required columns in the specified order
    result_df = df[['Voltage (V)', 'd (A)', 'd_error (A)', 'strain (%)', 'error_strain (%)', 'd_reference (A)']]
    # Save to a .data file (tab-separated text file)
    df.to_csv(f'{peak}_{Temperature}_strain.csv', sep='\t', index=False, float_format='%.6f')
    return str(f'{peak}_{Temperature}_strain.csv')

def plot_data(data_file, peak, Temperature, degree = 1):
    # Read the tab-separated CSV file
    df = pd.read_csv(data_file, sep='\t')
    
    # Extract data columns
    voltage = df['Voltage (V)']
    strain = df['strain (%)']
    error_strain = df['error_strain (%)']
    
    # Create scatter plot with error bars
    plt.errorbar(voltage, strain, yerr=error_strain, fmt='o', label='Data')
    
    # Fit a polynomial of specified degree
    coefficients = np.polyfit(voltage, strain, degree)
    p = np.poly1d(coefficients)
    
    # Generate points for the fitted curve
    x_fit = np.linspace(min(voltage), max(voltage), 100)
    y_fit = p(x_fit)
    
    # Plot the fitted polynomial
    plt.plot(x_fit, y_fit, label=f'Fitted polynomial (degree={degree})')
    
    # Construct the polynomial equation string
    eq_str = 'y = '
    for i, coef in enumerate(coefficients[::-1]):
        if i == 0:
            eq_str += f'{coef:.3f}'
        elif i == 1:
            eq_str += f' + {coef:.3f} x'
        else:
            eq_str += f' + {coef:.3f} x^{i}'
    eq_str = eq_str.replace('+ -', '- ')
    
    # Display the equation on the plot
    plt.text(0.05, 0.95, eq_str, transform=plt.gca().transAxes, fontsize=10, verticalalignment='top')
    
    # Set title and axis labels
    plt.title(f'Calibration Plot for {peak} at {Temperature}')
    plt.xlabel('Voltage (V)')
    plt.ylabel('Strain (%)')
    
    # Add legend
    plt.legend()
    
    # Display the plot
    plt.show()
    
    f, axs = plt.subplots(1, 2, figsize=(10, 5))  # Create a figure with two subplots

    # Print the fitted coefficients
    return print(f'Fitted polynomial coefficients (from highest to lowest degree): {coefficients}')

## Bragg Peak

# Track bragg peaks 3 0 0
Peak = "(3 0 0)"    # Peak
peak_hkl = (3, 0, 0)     # HKL value of the peak
peak_xy = [82,389]   # Pixel position in the raw data
peak_dx, peak_dy = 20, 20 # Box size for gathering data
N_pts = 52
Frame_margin = 10
do_update_xy = False

running_script_path = os.getcwd()

poni_file = 'LaB6_Ka_tth0.poni'
Z_dir = "/mnt/z/VEGA/CsV3Sb5_strain/2024/07/CsV3Sb5_July24_Kalpha/runs"
T = "15K"
out_file_name = f'{Peak}_{T}_Dvalues.csv'

get_data_and_plot(peak_xy, peak_dx, peak_dy, N_pts, Frame_margin, poni_file, Z_dir, T, running_script_path, out_file_name)

processed_data_file_name = process_data(out_file_name, Peak, T)
#processed_data_file_name = f'{Peak}_{T}_strain.csv'

plot_data(processed_data_file_name, Peak, T)
















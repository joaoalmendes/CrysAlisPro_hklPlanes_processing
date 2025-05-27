#!/usr/bin/env python
# coding: utf-8

import csv
import glob
import os
import fabio

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pyFAI import load
from scipy.ndimage import center_of_mass
from lmfit.models import PseudoVoigtModel, LinearModel

from huberutils.cbfTools import HuberCBFRun, libregex
from huberutils.fit import Kalpha_fit
from huberutils.utils import get_safe_slice

def Synch_peak_fit(x, y, wavelength):
    '''
    fits the data with the sum of Pseudovoigt and linear functions, supposing 
    that synchrotron monochromatic radiation was used

    function parameters
    -------------------
    x - independent variable in data
    y - dependent variable in data
    wavelength - the wavelength used for the experiment
    '''
    pv = PseudoVoigtModel()
    lm = LinearModel()
    model = pv + lm
    params = model.make_params()
    params.update(lm.guess(y, x))
    params.add("d", expr=f"{wavelength}/(2*sin(center/2*pi/180))")
    params.add("q", expr=f"4*pi/{wavelength}*sin(center/2*pi/180)")
    params.update(PseudoVoigtModel().guess(y - lm.eval(x=x, params=lm.guess(y, x)), x))
    res = model.fit(y, params=params, x=x)
    return res

def p_to_point(str):
    if 'P' in str:
        newstr = str.replace('P','.')
    else:
        newstr = str
    return newstr

def get_external_parameters(name):
    ext_par = {
        'Temperature': 300,
        'Pressure': 0,
        'Voltage': 0,
        'Capacity': 0
    }
    try:

        temp = float(name.upper().split("_T_")[1].split("_K")[0])
        ext_par['Temperature'] = temp
    except:
        pass
    try:
        pres = float(p_to_point(name.upper().split("_P_")[1].split("_GPA")[0]))
        ext_par['Pressure'] = pres
    except:
        pass
    try:
        #volt = float(name.upper().split("_U_")[1].split("_V")[0]) #WRONG
        # STRAIN DATA
        volt = float(name.split("/")[-1].split("_")[0][1:])
        ext_par['Voltage'] = volt
    except:
        pass
    try:
        cap = float(name.upper().split("_C_")[1].split("_MKF")[0])
        ext_par['Capacity'] = cap
    except:
        pass
    return ext_par

def get_voltage(name):
    '''
    gets the voltage from the file name

    function parameter
    ------------------
    name - filename
    '''
    return float(name.split("/")[-1].split("_")[0][1:])

def get_temperature(name):
    '''
    Gets the temperature from the filename
    '''
    return float(name.split("_t_")[1].split("k")[0])

def get_frame_index(fn):
    '''
    gets the frame index in the scan

    function parameter
    ------------------
    fn - frame filename
    '''
    return int(libregex.runre_soft.match(fn)["frame"])

def creating_dataset(basename):
    '''
    creates dataset from cbf files for further processing.
    replaces HuberRun in case the data were obtained not at the lab instruments
    '''
    frames = [fabio.open(_) for _ in np.sort(glob.glob(basename + "*.cbf"))]
    return np.ma.masked_less(np.array([_.data for _ in frames]), 0) 

def center_peak(run, frame, ij, thrng=3, dx=20, dy=20, _depth=0, _maxdepth=100):
    '''
    locates the peak center at the frame and through the frames
    and corrects corresponding coordinates i,j and frame index if needed
    returns ij coordinates and the frame number

    function parameters
    -------------------
    run - run object of the scan
    frame - frame index
    ij - peak coordinates at the frame [pixels]
    thrng=3 - a number of the neighbouring frames to be checked for the current peak (one side), thrng=3 means frame+-3 frames to be checked
    dx=20,dy=20 - size of the rectangle around ij where the center of mass should be determined [pixels]
    _depth - counter for the recurssion
    _maxdepth=100 - limit for recentering iterations
    '''
    center = [frame] + list(ij)
    deltas = [thrng, dx, dy]
    dat, origin, sll = get_safe_slice(run.data, center, deltas)
    new_frame, i, j = [int(np.round(_)) for _ in center_of_mass(dat) + origin]
    if (new_frame != frame) or (i != ij[0]) or (j != ij[1]) and (_depth < _maxdepth):
        return center_peak(run, new_frame, [i, j], thrng, dx, dy, _depth + 1)
    return (frame, ij)

def center_peak_data(dataset, frame, ij, thrng=3, dx=20, dy=20, _depth=0, _maxdepth=100):
    '''
    locates the peak center at the frame and through the frames
    and corrects corresponding coordinates i,j and frame index if needed
    returns ij coordinates and the frame number

    function parameters
    -------------------
    dataset - set of the data (usually 3D with x, y and frame dimensions)
    frame - frame index
    ij - peak coordinates at the frame [pixels]
    thrng=3 - a number of the neighbouring frames to be checked for the current peak (one side), thrng=3 means frame+-3 frames to be checked
    dx=20,dy=20 - size of the rectangle around ij where the center of mass should be determined [pixels]
    _depth - counter for the recurssion
    _maxdepth=100 - limit for recentering iterations
    '''
    center = [frame] + list(ij)
    deltas = [thrng, dx, dy]
    dat, origin, sll = get_safe_slice(dataset, center, deltas)
    new_frame, i, j = [int(np.round(_)) for _ in center_of_mass(dat) + origin]
    if (new_frame != frame) or (i != ij[0]) or (j != ij[1]) and (_depth < _maxdepth):
        return center_peak(dataset, new_frame, [i, j], thrng, dx, dy, _depth + 1)
    return (frame, ij)


    #!!! get_safe_slice should be rewritten since it counts frames from 0,
    # while in many datasets it starts with 0. Thus, you have 1 frame shift
    # in the center    # 

def center_peak_general(synchFlag, dataset, frame, ij, thrng=3, dx=20, dy=20, _depth=0, _maxdepth=100):
    '''
    locates the peak center at the frame and through the frames
    and corrects corresponding coordinates i,j and frame index if needed
    returns ij coordinates and the frame number

    function parameters
    -------------------
    dataset - set of the data (usually 3D with x, y and frame dimensions)
    frame - frame index
    ij - peak coordinates at the frame [pixels]
    thrng=3 - a number of the neighbouring frames to be checked for the current peak (one side), thrng=3 means frame+-3 frames to be checked
    dx=20,dy=20 - size of the rectangle around ij where the center of mass should be determined [pixels]
    _depth - counter for the recurssion
    _maxdepth=100 - limit for recentering iterations
    '''
    center = [frame] + list(ij)
    deltas = [thrng, dx, dy]
    if synchFlag:
        dat, origin, sll = get_safe_slice(dataset, center, deltas)
    else:
        dat, origin, sll = get_safe_slice(dataset.data, center, deltas)
    new_frame, i, j = [int(np.round(_)) for _ in center_of_mass(dat) + origin]
    if (new_frame != frame) or (i != ij[0]) or (j != ij[1]) and (_depth < _maxdepth):
        return center_peak_general(synchFlag, dataset, new_frame, [i, j], thrng, dx, dy, _depth + 1)
    return (frame, ij)


    #!!! get_safe_slice should be rewritten since it counts frames from 0,
    # while in many datasets it starts with 1. Thus, you have 1 frame shift
    # in the center    # 

def process_peak(
    run, frame, ij, ai, center=True, pnts_fac=5, fit=Kalpha_fit, thrng=3, dx=10, dy=10
):
    '''
    processes the peak returns its properties like
    Isum - total intensity [arb. un.]
    tth - 2theta position [degree]
    Itth - intensity at tth
    th - theta position [degree]
    Ith - intensity at th [arb. un.]
    frame - frame index
    ij - ccordinates of the peak center [pixels]
    res - contains fit parameters

    function parameters
    -------------------
    run - run object of the scan
    frame - center of the peak frame index
    ij - canter of the pek coordinates at the frame [pixels]
    ai - pyFai azimuthal integration object
    center=True - flag for centering the peak
    pnts_fac=5
    fit=Kalpha_fit - function used for 2theta profile fitting. Kalpha_fit corresponds to two pseudovoight with the fixed peak ratios
    thrng=3 - theta range
    dx=10, dy=10 - size of the rectanle around the peak to be analysed
    
    
    '''
    ai.rot2 = run.th2 * np.pi / 180
    s_wl = ai.wavelength
    mask = np.ones(run.frames[0].data.shape)
    if center:
        frame, ij = center_peak(run, frame, ij)
    mask[ij[0] - dx : ij[0] + dx + 1, ij[1] - dy : ij[1] + dy + 1] = 0
    dat, origin, sll = get_safe_slice(run.data, [frame] + list(ij), [thrng, dx, dy])

    tth, Itth = ai.integrate1d(
        run.data[frame - 1 : frame + 2].sum(axis=0),
        npt=pnts_fac * dx,
        unit="2th_deg",
        mask=mask,
    )
    th = (run.th_s + 0.5 * run.th_i)[sll[0]]
    Ith = run.data[sll].sum(axis=(1, 2))
    print(run.data[sll].shape)
    Isum = Ith.sum()

    # ! K_alpha function should be rewritten in order to read wavelength

    res = fit(tth, Itth)
    return (Isum, tth, Itth, th, Ith, frame, ij, res)

def process_peak_dataset(
    dataset, frame, ij, ai, center=True, pnts_fac=5, fit=Synch_peak_fit, thrng=3, dx=10, dy=10
):
    '''
    processes the peak returns its properties like
    Isum - total intensity [arb. un.]
    tth - 2theta position [degree]
    Itth - intensity at tth
    th - theta position [degree]
    Ith - intensity at th [arb. un.]
    frame - frame index
    ij - ccordinates of the peak center [pixels]
    res - contains fit parameters

    function parameters
    -------------------
    dataset - set of the data (usually 3D with x, y and frame dimensions)
    frame - center of the peak frame index
    ij - canter of the pek coordinates at the frame [pixels]
    ai - pyFai azimuthal integration object
    center=True - flag for centering the peak
    pnts_fac=5
    fit=Kalpha_fit - function used for 2theta profile fitting. Kalpha_fit corresponds to two pseudovoight with the fixed peak ratios
    thrng=3 - theta range
    dx=10, dy=10 - size of the rectanle around the peak to be analysed
    
    
    '''
    #ai.rot2 = run.th2 * np.pi / 180
    s_wl = ai.wavelength
    mask = np.ones(dataset[0].shape)
    if center:
        frame, ij = center_peak_data(dataset, frame, ij)
    mask[ij[0] - dx : ij[0] + dx + 1, ij[1] - dy : ij[1] + dy + 1] = 0
    dat, origin, sll = get_safe_slice(dataset, [frame] + list(ij), [thrng, dx, dy])

    tth, Itth = ai.integrate1d(
        dataset[frame - 1 : frame + 2].sum(axis=0),
        npt=pnts_fac * dx,
        unit="2th_deg",
        mask=mask,
    )
    # th = (run.th_s + 0.5 * run.th_i)[sll[0]]
    th = (th_range+0.5*0.1)[sll[0]]
    Ith = dataset[sll].sum(axis=(1, 2))
    print(dataset[sll].shape)
    Isum = Ith.sum()

    res = fit(tth, Itth, s_wl)
    return (Isum, tth, Itth, th, Ith, frame, ij, res)

def process_peak_general(synchFlag,
    dataset, frame, ij, ai, center=True, pnts_fac=5, thrng=3, dx=10, dy=10
):
    '''
    processes the peak returns its properties like
    Isum - total intensity [arb. un.]
    tth - 2theta position [degree]
    Itth - intensity at tth
    th - theta position [degree]
    Ith - intensity at th [arb. un.]
    frame - frame index
    ij - ccordinates of the peak center [pixels]
    res - contains fit parameters

    function parameters
    -------------------
    dataset - set of the data (usually 3D with x, y and frame dimensions)
    frame - center of the peak frame index
    ij - canter of the pek coordinates at the frame [pixels]
    ai - pyFai azimuthal integration object
    center=True - flag for centering the peak
    pnts_fac=5
    fit=Kalpha_fit - function used for 2theta profile fitting. Kalpha_fit corresponds to two pseudovoight with the fixed peak ratios
    thrng=3 - theta depth
    dx=10, dy=10 - size of the rectanle around the peak to be analysed
    
    
    '''
    if synchFlag:
        working_data = dataset
        fit = Synch_peak_fit
        s_wl = ai.wavelength
        mask = np.ones(dataset[0].shape)


    else:
        working_data = dataset.data
        fit = Kalpha_fit
        ai.rot2 = dataset.th2 * np.pi / 180
        mask = np.ones(dataset.frames[0].data.shape)


    #s_wl = ai.wavelength
    
    if center:
        frame, ij = center_peak_general(synchFlag, dataset, frame, ij)
    mask[ij[0] - dx : ij[0] + dx + 1, ij[1] - dy : ij[1] + dy + 1] = 0
    dat, origin, sll = get_safe_slice(working_data, [frame] + list(ij), [thrng, dx, dy])

    if synchFlag:
        th = (th_range+0.5*0.1)[sll[0]]
    else:
        th = (dataset.th_s + 0.5 * dataset.th_i)[sll[0]]
    Ith = working_data[sll].sum(axis=(1, 2))


    tth, Itth = ai.integrate1d(
        working_data[frame - 1 : frame + 2].sum(axis=0),
        npt=pnts_fac * dx,
        unit="2th_deg",
        mask=mask,
    )

    print(working_data[sll].shape)
    Isum = Ith.sum()

    # ! K_alpha function should be rewritten in order to read wavelength
    res = fit(tth, Itth)
    return (Isum, tth, Itth, th, Ith, frame, ij, res)

def prep_results_from_fit(res):
    '''
    reads the fit results

    function parameter
    -------------------
    
    res - results of the I(2theta) fit with the fitting function (by default Kalpha_fit) 
    
    '''
    out = []
    for prop in ["d", "center", "amplitude"]:
        out.append(res.params[prop].value)
        out.append(res.params[prop].stderr if res.params[prop].stderr else -1)
    return out


def save_xy(xy, fn):
    '''
    saves the data as x y textfile

    function parameters
    -------------------
    xy - x y data
    fn - file name
    '''
    with open(fn, "w") as fh:
        for x, y in zip(*xy):
            fh.write("{:15.8f} {:15.8f}\n".format(x, y))

poni_file = 'LaB6_Ka_tth0.poni'
Z_dir = "/mnt/z/VEGA/CsV3Sb5_strain/2024/07/CsV3Sb5_July24_Kalpha/runs"
script_dir = os.getcwd()
T = "80K"

if T == "80K":
    peak_dir = "6peaks"
    end = "merged"
elif T == "15K":
    peak_dir = "7peaks"
    end = "esp"
else:
    raise ValueError("Invalid temperature. Use '80K' or '15K'.")

# csv-file with the peak table
peak_table = os.path.join(script_dir, f"{T}_peak_table.csv")
# flag, showing whether the measurement was performed in the lab or at synchrotron
isAtSynch = False
# range of the angles th in one measurement
th_range = np.arange(-10,10,0.1)
# template of the folder name where the dataset of one external parameter point is stored
folder_name_template = os.path.join(Z_dir, T, "V*_*", peak_dir, end)
# path to the appropriate poni-file
poni_file = os.path.join(Z_dir, T, poni_file)
# tail of the *.cbf file names
if T == "80K":
    file_name_tail = "CsV3Sb5_strain_merged_*.cbf"     
if T == "15K":
    file_name_tail = "CsV3Sb5_strain_*.cbf"     
# A main external parameter
ext_param = 'Voltage'

reader = csv.reader(open(peak_table))
ijs = [] # list for the i, j coordinates of the peak maximum [pixels] 
hkls = [] # list for the peak's hkl triades
fns = [] # list for the frame files where the peak is detected

for row in reader:
    ijs.append([int(_) for _ in row[:2]][::-1])
    hkls.append([float(_) for _ in row[-3:]])
    fns.append(row[2])

def main():
    
    head = glob.glob(folder_name_template) # getting the list of folders with data files
    head.sort(key=lambda f: get_external_parameters(f)[ext_param])
    print('The next folders will be analysed:')
    print(*head, sep='\n')
    ai = load(poni_file)
 
    
    pi = 0 # peak index
    for ij, hkl, fn in zip(ijs, hkls, fns):
        hkl_string = (3 * "_{:.2f}").format(*hkl)
        res_dir = f"{script_dir}/results/{pi:02}{hkl_string}"
        print('\nResults of the analysis will be stored in\n', res_dir,'\n')
        plot_dir = res_dir + f"/plots{hkl_string}"
        data_dir = res_dir + f"/data{hkl_string}"
        os.makedirs(plot_dir, exist_ok=True)
        os.makedirs(data_dir, exist_ok=True)
        fout = open(res_dir + f"/peak_dep{hkl_string}.csv", "w")
        fout.write(
                f'{ext_param}\tIsum\tdIsum\td\tdd\t2theta\td2theta\tAmplitude\tdAmplitude\n'
            )
        for i, pre in enumerate(head):
            
            EP = get_external_parameters(pre)[ext_param]
            print('EP', EP)

            # SHOULD BE REWRITTEN!!!
            #tail = f"pbcute2o6_1007_800mm_atten_2_t_{int(EP)}k_00001_"
            #if EP == EP: # just for testing
            #if EP > 1.5: # test for STRAIN DATA
            #    continue
            print(f"loading data for {ext_param[0]} = {EP}")
            #print('os.path.join(pre, file_name_tail)', os.path.join(pre, file_name_tail))
            try:
                if isAtSynch:
                    dataset = creating_dataset(os.path.join(pre, file_name_tail))
                else:
                    dataset = HuberCBFRun(os.path.join(pre, file_name_tail))
                    #print('dataset', dataset)
                    dataset.th_s = np.array(
                        [dataset.th_s[0] + dataset.th_i[0]*_ for _ in range(len(dataset.frames))])
                        
            except:
                
               print('Couldn\'t load the files. Please check the format')
               print(Exception)
               continue

            print(f"analysing peak {pi:02} at {ext_param[0]} = {EP} ... ")
            frame = get_frame_index(fn)
            print(frame)
            Isum, tth, Itth, th, Ith, frame_new, ij_new, res = process_peak_general(
                isAtSynch, dataset, frame, ij, ai
            )
            print(f"update peak position from {ij} to {ij_new}!")
            ij = ij_new
            print(f"update frame from {frame} to {frame_new}!")
            frame = frame_new
            
            fout.write(
                (9 * "{:15.8f} ").format(
                    EP, Isum, np.sqrt(Isum), *prep_results_from_fit(res)
                )
                + "\n"
            )
            save_xy([tth, Itth], data_dir + f"/{i:02}_{ext_param[0]} = {EP}_tth_I.txt")
            save_xy([th, Ith], data_dir + f"/{i:02}_{ext_param[0]} = {EP}_th_I.txt")
            fig, axs = plt.subplots(1, 2, constrained_layout=True)
            res.plot_fit(title=f"{ext_param[0]} = {EP} " + hkl_string, ax=axs[0])
            axs[1].plot(th, Ith, "bo-")
            axs[1].set_xlabel("Theta")
            axs[0].set_xlabel("2Theta")
            for a in axs:
                a.set_ylabel("Intensity")
            plt.savefig(plot_dir + f"/{i:02}_{ext_param[0]} = {EP}_fit.pdf")
            plt.close()
            print("done!")
        fout.close()
        pi += 1
    print(10*'-'+' The analysis is complete '+10*'-')

if __name__ == "__main__":
    main()
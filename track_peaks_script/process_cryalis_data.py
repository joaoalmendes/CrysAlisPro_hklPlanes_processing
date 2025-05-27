import os
import glob
import fabio
from skimage.feature import peak_local_max
from scipy.ndimage import median_filter
import numpy as np
import pandas as pd
from pyFAI import load as load_poni

def detect_peaks(data, min_distance=25, threshold_rel=0.50):
    """Detect peaks in a 2D image array."""
    filtered = median_filter(data, size=20)
    peaks = peak_local_max(filtered, min_distance=min_distance, threshold_rel=threshold_rel)
    return [{'i': int(i), 'j': int(j)} for i, j in peaks]

def parse_runlist(runlist_file):
    """Parse the runlist.txt file to extract run parameters."""
    runs = []
    with open(runlist_file, 'r') as f:
        for line in f:
            if line.startswith('#RS#'):
                continue
            if line.startswith('#RE#'):
                break
            parts = line.split()
            if len(parts) >= 10:
                run_num = int(parts[0])
                th_s = float(parts[1])
                th_inc = float(parts[2])
                nframes = int(parts[3])
                runs.append({
                    'run_num': run_num,
                    'th_s': th_s,
                    'th_inc': th_inc,
                    'nframes': nframes
                })
    return runs

def compute_q(j, i, ai, omega_rad, pixel_size=0.000172):
    """Compute the q-vector for a pixel (j, i) in Å⁻¹."""
    # PONI parameters
    D = ai.dist  # Sample-to-detector distance (m)
    poni1 = ai.poni1  # Vertical beam center (m)
    poni2 = ai.poni2  # Horizontal beam center (m)
    wavelength = ai.wavelength * 1e10  # Convert meters to Å (7.0936e-11 m = 0.70936 Å)
    rot2 = ai.rot2  # Detector tilt (radians)
    
    # Convert pixel coordinates to physical distances (in meters)
    j_center = poni2 / pixel_size  # Beam center in pixels (horizontal)
    i_center = poni1 / pixel_size  # Beam center in pixels (vertical)
    x = (j - j_center) * pixel_size
    y = (i - i_center) * pixel_size
    
    # Calculate scattering angles
    r = np.sqrt(x**2 + y**2)
    theta2 = np.arctan2(r, D)
    chi = np.arctan2(y, x)
    
    # q magnitude in Å⁻¹
    q_magnitude = (4 * np.pi / wavelength) * np.sin(theta2 / 2)
    
    # q in detector coordinates (before detector tilt)
    q_det = q_magnitude * np.array([
        np.cos(chi) * np.sin(theta2),
        np.sin(chi) * np.sin(theta2),
        np.cos(theta2)
    ])
    
    # Apply detector tilt (Rot2 around y-axis)
    cos_rot2 = np.cos(rot2)
    sin_rot2 = np.sin(rot2)
    R_rot2 = np.array([
        [cos_rot2, 0, sin_rot2],
        [0, 1, 0],
        [-sin_rot2, 0, cos_rot2]
    ])
    q_det = np.dot(R_rot2, q_det)
    
    # Rotation matrix for omega (rotation around y-axis)
    cos_omega = np.cos(omega_rad)
    sin_omega = np.sin(omega_rad)
    R_omega = np.array([
        [cos_omega, 0, sin_omega],
        [0, 1, 0],
        [-sin_omega, 0, cos_omega]
    ])
    
    # Apply sample rotation
    q = np.dot(R_omega, q_det)
    
    return q

# Base directory
Z_DIR = "/mnt/z/VEGA/CsV3Sb5_strain/2024/07/CsV3Sb5_July24_Kalpha/runs"
SCRIPT_DIR = os.getcwd()

# Get temperature input
temp = "80K"  # Hardcoded for testing
if temp == "80K":
    peak_dir = "6peaks"
elif temp == "15K":
    peak_dir = "7peaks"
else:
    raise ValueError("Invalid temperature. Use '80K' or '15K'.")

# Maximum absolute hkl value for filtering (set to float('inf') to disable)
max_hkl = 5  # Changeable for testing; set to float('inf') to include all peaks

# Find all voltage folders for the given temperature
voltage_folders = glob.glob(os.path.join(Z_DIR, temp, "V*_*/"))
if not voltage_folders:
    raise FileNotFoundError(f"No voltage folders found in {os.path.join(Z_DIR, temp)}")

# Load poni and UB matrix
poni_file = os.path.join(Z_DIR, temp, "LaB6_Ka_tth0.poni")
ub_file = os.path.join(SCRIPT_DIR, "ub_matrix_input.txt")
ai = load_poni(poni_file)
UB = np.loadtxt(ub_file).reshape(3, 3)
UB_inv = np.linalg.inv(UB)

# Process each voltage folder
peak_data = []
for voltage_folder in voltage_folders:
    voltage = os.path.basename(os.path.dirname(voltage_folder))
    print(f"Processing voltage folder: {voltage}")
    
    # Check if peak directory exists
    peak_path = os.path.join(voltage_folder, peak_dir)
    if not os.path.exists(peak_path):
        continue
    
    # Load runlist.txt from peak directory
    runlist_file = os.path.join(peak_path, "runlist.txt")
    if not os.path.exists(runlist_file):
        print(f"Skipping {voltage}: runlist.txt not found in {peak_path}")
        continue
    runs = parse_runlist(runlist_file)
    
    # Find CBF files in merged folder
    merged_path = os.path.join(peak_path, "merged")
    if temp == "35K":
        merged_path = os.path.join(peak_path, "esp")
    if not os.path.exists(merged_path):
        print(f"Skipping {voltage}: merged folder not found in {merged_path}")
        continue
    cbf_files = glob.glob(os.path.join(merged_path, "CsV3Sb5_strain_merged_*.cbf"))
    if temp == "35K":
        cbf_files = glob.glob(os.path.join(merged_path, "CsV3Sb5_strain_*.cbf"))
    if not cbf_files:
        print(f"Skipping {voltage}: No CBF files found in {merged_path}")
        continue
    
    # Dictionary to track peaks with max intensity per hkl
    peak_dict = {}
    
    # Process each CBF file
    for file in cbf_files:
        basename = os.path.basename(file)
        #print(f"{voltage}: {basename}")
        parts = basename.split('_')
        if len(parts) < 4:
            print(f"Skipping malformed filename: {basename}")
            continue
        try:
            run = int(parts[-2])
            frame = int(parts[-1].split('.')[0])
        except (ValueError, IndexError):
            print(f"Skipping file with invalid naming: {basename}")
            continue
        run_info = next((r for r in runs if r['run_num'] == run), None)
        if run_info is None:
            print(f"Warning: No run info for run {run} in {voltage}")
            continue
        th_s = run_info['th_s']
        th_inc = run_info['th_inc']
        omega = th_s + (frame - 1) * th_inc # in degrees
        omega_rad = np.deg2rad(omega)
        img = fabio.open(file)
        data = img.data
        peaks = detect_peaks(data)
        for peak in peaks:
            try:
                # Calculate q-vector in Å⁻¹
                q = compute_q(peak['j'], peak['i'], ai, omega_rad)
                if q is None:
                    raise ValueError("Failed to calculate q-vector")
                hkl = np.dot(UB_inv, q)
                hkl_rounded = np.round(hkl, 2)
                # Filter peaks with |h|, |k|, |l| > max_hkl
                if any(abs(hkl_rounded[i]) > max_hkl for i in range(3)):
                    continue
                # Get peak intensity at (i, j)
                intensity = data[peak['i'], peak['j']]
                
                # Check if peak matches an existing hkl (within 0.15 tolerance)
                hkl_tuple = tuple(hkl_rounded)
                matched = False
                for existing_hkl in list(peak_dict.keys()):
                    if all(abs(hkl_rounded[i] - existing_hkl[i]) <= 0.1 for i in range(3)):
                        # Same peak; update if intensity is higher
                        if intensity > peak_dict[existing_hkl]['intensity']:
                            peak_dict[existing_hkl] = {
                                'intensity': intensity,
                                'j': peak['j'],
                                'i': peak['i'],
                                'frame': frame,
                                'cbf_file': basename,
                                'hkl': hkl_rounded
                            }
                        matched = True
                        break
                if not matched:
                    # New peak
                    peak_dict[hkl_tuple] = {
                        'intensity': intensity,
                        'j': peak['j'],
                        'i': peak['i'],
                        'frame': frame,
                        'cbf_file': basename,
                        'hkl': hkl_rounded
                    }
            except Exception as e:
                print(f"Error processing peak in frame {frame}, run {run}, voltage {voltage}: {e}")
    #print(peak_dict)
    # Append max-intensity peaks for this voltage to peak_data
    for peak_info in peak_dict.values():
        peak_data.append([
            peak_info['j'],  # i/x
            peak_info['i'],  # j/y
            peak_info['cbf_file'],  # data frame
            peak_info['hkl'][0],  # h
            peak_info['hkl'][1],  # k
            peak_info['hkl'][2]   # l
        ])

# Save results to CSV with specified column order
if peak_data:
    df = pd.DataFrame(peak_data, columns=['i/x', 'j/y', 'data frame', 'h', 'k', 'l'])
    output_file = os.path.join(SCRIPT_DIR, f'{temp}_peak_table.csv')
    df.to_csv(output_file, index=False)
    print(f"Saved peak data to {output_file}")
else:
    print("No peaks detected or processed.")
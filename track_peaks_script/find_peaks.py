import os
import glob
import fabio
import numpy as np
import pandas as pd
from pyFAI import load as load_poni

# Helper functions
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

def compute_pixel_from_hkl(hkl, omega_rad, ai, pixel_size):
    """Compute the pixel position (j, i) for a given hkl and omega."""
    hkl = np.array(hkl)
    # Compute R_omega_inv (rotation matrix for -omega)
    cos_omega = np.cos(-omega_rad)
    sin_omega = np.sin(-omega_rad)
    R_omega_inv = np.array([
        [cos_omega, 0, sin_omega],
        [0, 1, 0],
        [-sin_omega, 0, cos_omega]
    ])
    # Compute q_sample = UB * hkl
    q_sample = np.dot(UB, hkl)
    # Compute q_det = R_omega_inv * q_sample
    q_det = np.dot(R_omega_inv, q_sample)
    # Compute R_rot2_inv (inverse detector tilt rotation)
    rot2 = ai.rot2
    cos_rot2 = np.cos(-rot2)
    sin_rot2 = np.sin(-rot2)
    R_rot2_inv = np.array([
        [cos_rot2, 0, sin_rot2],
        [0, 1, 0],
        [-sin_rot2, 0, cos_rot2]
    ])
    # Compute q_det_untilted = R_rot2_inv * q_det
    q_det_untilted = np.dot(R_rot2_inv, q_det)
    # Compute q_magnitude
    q_magnitude = np.linalg.norm(q_det_untilted)
    if q_magnitude == 0:
        return None
    # Get wavelength in Ångströms (ai.wavelength is in meters)
    wavelength = ai.wavelength * 1e10  # Convert meters to Å
    # Compute theta2 using q = (4π/λ) * sin(θ2/2)
    sin_theta_half = q_magnitude * wavelength / (4 * np.pi)
    if abs(sin_theta_half) > 1:
        return None
    theta2 = 2 * np.arcsin(sin_theta_half)
    # Compute chi
    chi = np.arctan2(q_det_untilted[1], q_det_untilted[0])
    # Compute r, x, y
    D = ai.dist
    r = D * np.tan(theta2)
    x = r * np.cos(chi)
    y = r * np.sin(chi)
    # Compute pixel coordinates
    j_center = ai.poni2 / pixel_size
    i_center = ai.poni1 / pixel_size
    j = j_center + x / pixel_size
    i = i_center + y / pixel_size
    return j, i

# Define constants
Z_DIR = "/mnt/z/VEGA/CsV3Sb5_strain/2024/07/CsV3Sb5_July24_Kalpha/runs"
SCRIPT_DIR = os.getcwd()
temp = "15K"  # Hardcoded for testing; can be "80K" or "15K"
if temp == "80K":
    peak_dir = "6peaks"
elif temp == "15K":
    peak_dir = "7peaks"
else:
    raise ValueError("Invalid temperature. Use '80K' or '15K'.")
pixel_size = 0.000172  # meters; this iformation comes from the poni file itself
dx, dy = 20, 20  # box size in pixels
background_factor = 2.0  # threshold factor for peak detection; this gives a 2\sigma certainty

# Load input hkl from CSV
input_file = os.path.join(SCRIPT_DIR, "input_find_peaks.csv")
df_input = pd.read_csv(input_file)
hkl_list = [tuple(row) for row in df_input[['h', 'k', 'l']].to_numpy()]

# Load poni and UB matrix
poni_file = os.path.join(Z_DIR, temp, "LaB6_Ka_tth0.poni")
ub_file = os.path.join(SCRIPT_DIR, "ub_matrix_input.txt")
ai = load_poni(poni_file)
UB = np.loadtxt(ub_file).reshape(3, 3)

# Initialize peak_dict to track maximum intensity for each hkl
peak_dict = {hkl: {'intensity': 0, 'j': None, 'i': None, 'cbf_file': None} for hkl in hkl_list}

# Find all voltage folders for the given temperature
voltage_folders = glob.glob(os.path.join(Z_DIR, temp, "V*_*/"))
for voltage_folder in voltage_folders:
    voltage = os.path.basename(os.path.dirname(voltage_folder))
    print(f"Processing voltage folder: {voltage}")
    
    peak_path = os.path.join(voltage_folder, peak_dir)
    if not os.path.exists(peak_path):
        continue
    
    # Load runlist.txt from peak directory
    runlist_file = os.path.join(peak_path, "runlist.txt")
    if not os.path.exists(runlist_file):
        continue
    runs = parse_runlist(runlist_file)
    
    # Find CBF files in merged folder
    merged_path = os.path.join(peak_path, "merged/")
    if not os.path.exists(merged_path):
        continue
    if temp == "80K":
        cbf_files = glob.glob(os.path.join(merged_path, "CsV3Sb5_strain_merged_*.cbf"))
    elif temp == "15K":
        cbf_files = glob.glob(os.path.join(merged_path, "CsV3Sb5_strain_*.cbf"))
    for cbf_file in cbf_files:
        basename = os.path.basename(cbf_file)
        parts = basename.split('_')
        try:
            run = int(parts[-2])
            frame = int(parts[-1].split('.')[0])
        except (ValueError, IndexError):
            continue
        run_info = next((r for r in runs if r['run_num'] == run), None)
        if run_info is None:
            continue
        th_s = run_info['th_s']
        th_inc = run_info['th_inc']
        omega = th_s + (frame - 1) * th_inc  # in degrees
        omega_rad = np.deg2rad(omega)
        
        # Load CBF file with error handling
        try:
            img = fabio.open(cbf_file)
        except Exception as e:
            print(f"Error reading {cbf_file}: {e}")
            continue
        data = img.data
        height, width = data.shape
        background = np.median(data)
        
        # Compute pixel positions for each hkl
        for hkl in hkl_list:
            pixel = compute_pixel_from_hkl(hkl, omega_rad, ai, pixel_size)
            if pixel is None:
                intensity = 0
            else:
                j, i = pixel
                j_int = int(np.round(j))
                i_int = int(np.round(i))
                i_start = max(0, i_int - dy//2)
                i_end = min(height, i_int + dy//2 + 1)
                j_start = max(0, j_int - dx//2)
                j_end = min(width, j_int + dx//2 + 1)
                if i_start >= i_end or j_start >= j_end:
                    intensity = 0
                else:
                    data_sub = data[i_start:i_end, j_start:j_end]
                    intensity = np.sum(data_sub) / ((dy + 1) * (dx + 1))  # Average intensity in 21x21 box
                    if intensity < background_factor * background:
                        intensity = 0
            if intensity > peak_dict[hkl]['intensity']:
                peak_dict[hkl]['intensity'] = intensity
                peak_dict[hkl]['j'] = j
                peak_dict[hkl]['i'] = i
                peak_dict[hkl]['cbf_file'] = basename

# Create DataFrame from peak_dict
peak_data = []
for hkl, info in peak_dict.items():
    if info['intensity'] > 0:
        peak_data.append([hkl[0], hkl[1], hkl[2], info['j'], info['i'], info['cbf_file'], info['intensity']])
if peak_data:
    df = pd.DataFrame(peak_data, columns=['h', 'k', 'l', 'j', 'i', 'cbf_file', 'I'])
    output_file = os.path.join(SCRIPT_DIR, f'{temp}_peak_intensities.csv')
    df.to_csv(output_file, index=False)
    print(f"Saved peak data to {output_file}")
else:
    print("No peaks with intensity above threshold found.")


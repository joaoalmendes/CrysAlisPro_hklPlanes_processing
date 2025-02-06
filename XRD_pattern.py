import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from ase import Atoms
from ase import io
from ase.visualize import view
from ase.geometry import cellpar_to_cell
from mpl_toolkits.mplot3d import Axes3D

def model_positions_to_ASE_positions(model_positions):
    input_pos_list = []
    for i in model_positions.values():
        for j in i:
            input_pos_list.append(j)
    return input_pos_list

#def unit_cell(a, b, c, formula, atoms_positions):
    bulk = Atoms(
        formula,
        positions=atoms_positions,
        cell=[a, b, c, 90, 90, 90],
        pbc=True,
    )
    return bulk

def unit_cell(a, b, c, alpha, beta, gamma, formula, input_positions):
    cell_matrix = cellpar_to_cell([a, b, c, alpha, beta, gamma])
    atoms = Atoms(
        formula,
        positions=np.dot(input_positions, cell_matrix),
        cell=cell_matrix,
        pbc=True,
    )

    return atoms

def atomic_form_factor(atom, q):
    factors = {
        'Cs': lambda q: 1/(1+(q/50)**2)**2 ,    # Z = 55
        'V': lambda q: 1/(1+(q/23)**2)**2 , # Z = 23
        'Sb': lambda q:  1/(1+(q/46)**2)**2  # Z = 51
    }
    return factors[atom](q)

def F_hkl(atoms, G_vector, q_norm):    
    # Precompute atomic form factors for each unique atom type
    atom_types = np.unique(atoms.get_chemical_symbols())  # Get unique atom types
    form_factors = {atom: atomic_form_factor(atom, q_norm) for atom in atom_types}

    # Get the atomic positions and symbols
    positions = atoms.get_positions()
    symbols = atoms.get_chemical_symbols()

    # Vectorized computation of structure factor
    exp_terms = np.exp(-1j * 2 * np.pi * np.dot(positions, G_vector))  # Phase term for all atoms
    f_atoms = np.array([form_factors[symbol] for symbol in symbols])  # Form factors for all atoms
    # Compute the structure factor
    F_hkl = np.sum(f_atoms * exp_terms)  # Sum over all atoms
    return F_hkl

def shape_factor(hkl, dimensions):
    Nx, Ny, Nz = dimensions
    """Compute shape factor S(hkl) for a finite N-unit-cell crystal"""
    h, k, l = hkl
    def sinc(x):
        return 1 if x == 0 else np.sin(np.pi * x) / (np.pi * x)
    
    return sinc(Nx * h) * sinc(Ny * k) * sinc(Nz * l)

def precompute_rotation_matrices(angles_deg):
    """ Precompute the 2D rotation matrices for a list of angles using NumPy operations. """
    angles_deg = np.array(angles_deg)
    angles_rad = np.radians(angles_deg)  # Convert all angles to radians at once
    cos_vals = np.cos(angles_rad)
    sin_vals = np.sin(angles_rad)

    # Stack rotation matrices in a dictionary
    return {
        angle: np.array([[c, -s], [s, c]])
        for angle, c, s in zip(angles_deg, cos_vals, sin_vals)
    }

def rotate_reciprocal_lattice(hkl, angle_deg, rotation_matrices):
    """Rotate (h, k, l) using a precomputed rotation matrix with NumPy operations."""
    hkl_array = np.asarray(hkl)
    h_k_rot = np.dot(rotation_matrices[angle_deg], hkl_array[:2])  # Use np.dot()
    return np.hstack((h_k_rot, hkl_array[2]))  # Efficient concatenation

def LP_correction(q_norm, wavelength):
    argument = wavelength * q_norm / (4 * np.pi)
    
    # Clamp argument to avoid NaN in arcsin
    argument = np.clip(argument, -1 + 1e-10, 1 - 1e-10)
    
    theta = np.arcsin(argument)
    
    # Compute LP factor safely
    sin_theta_sq = np.sin(theta) ** 2
    cos_theta = np.cos(theta)
    
    # Avoid division by zero, replace with a large number
    LP_factor = np.where((sin_theta_sq > 1e-10) & (cos_theta > 1e-10),
                         1 / (sin_theta_sq * cos_theta),
                         np.inf)

    # Normalize: Map 0 → 0 and Inf → 1
    LP_normalized = 1 - (1 / (1 + LP_factor))

    return LP_normalized

def DW_correction(q, B = 0.001):
    return np.float64(np.exp(-B * q**2))

def I_hkl(atoms, hkl, r_lattice, twin_angles, twin_fractions, rotation_matrices, wavelength = 1.54):
    F_total = 0
    for twin_angle, twin_fraction in zip(twin_angles, twin_fractions):
        hkl_twin = rotate_reciprocal_lattice(hkl, twin_angle, rotation_matrices)
        G = np.dot(hkl_twin, r_lattice)
        q = np.linalg.norm(G)  # Magnitude of reciprocal lattice vector
        F_twin = F_hkl(atoms, G, q) * twin_fraction * DW_correction(q) * LP_correction(q, wavelength)
        F_total += F_twin
    return np.abs(F_total)**2

def plot_XRD_pattern(h_range, k_range, l_cuts, atoms, twin_angles, twin_fractions):
    recip_lattice = np.round(np.linalg.inv(atoms.get_cell().T) * (2 * np.pi), decimals=10)
    rotation_matrices = precompute_rotation_matrices(twin_angles)
    for l in l_cuts:
        h, k = np.meshgrid(h_range, k_range)
        intensity = np.zeros((len(h_range), len(k_range)))

        for i in range(len(h_range)):
            for j in range(len(k_range)):
                hkl = (h[i, j], k[i, j], l)
                hkl = tuple(np.round(hkl, decimals=10))
                intensity[i, j] = I_hkl(atoms, hkl, recip_lattice, twin_angles, twin_fractions, rotation_matrices)

        intensity /= np.max(intensity)  # Normalize
        intensity = gaussian_filter(intensity, sigma=1.5)  # Smooth
        intensity = np.log1p(intensity)  # Log scaling

        # Plot the intensity map
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111)

        # Plot the intensity as a heatmap
        c = ax.pcolormesh(h, k, intensity, shading='auto', cmap='inferno')
        fig.colorbar(c, label='Intensity')

        # Labels and title
        ax.set_xlabel('h')
        ax.set_ylabel('k')
        ax.set_title(f"Reciprocal Space Intensity (l={l})")
        #plt.show()
        plt.savefig(f"sim_hk{l}.png", dpi = 300)
        plt.close()
    return None

model_1_positions = {
    "Cs": [(0, 0, 0.750748), (0.25, 0.25, 0.25)],
    "V": [
        (0.3785912, 0.126306, 0),
        (0.6241512, 0.376246, 0),
        (0.24752, 0, 0),
        (0, 0.7474110, 0),
    ],
    "Sb": [
        (0, 0.332945, 0.6215610),
        (0.250563, 0.083484, 0.121629),
        (0, 0.833395, 0.6233810),
        (0, 0, 0.5),
        (0.25, 0.25, 0),
        (0, 0, 0),
    ],
}

model_2_positions = {
    "Cs": [(0, 0, 0.749239), (0.25, 0.25, 0.25)],
    "V": [
        (0.3715413, 0.123737, 0),
        (0.6258714, 0.373747, 0),
        (0.25252, 0, 0),
        (0, 0.7525912, 0),
    ],
    "Sb": [
        (0, 0.333695, 0.6225110),
        (0.249444, 0.083175, 0.122479),
        (0, 0.833275, 0.620711),
        (0, 0, 0.5),
        (0.25, 0.25, 0),
        (0, 0, 0),
    ],
}

# Initial undistorted lattice constants (Å)
a, b, c = 10.971314, 18.9833, 18.51410  
alpha, beta, gamma = 90, 90, 90
formula = "Cs2V4Sb6"
atom_types = ['Cs', 'V', 'Sb']  # Atom types for each position
bulk_dimensions = (3, 3, 10)
twin_angles, twin_populations = [0, 120, 240], [np.float64(0.45), np.float64(0.05), np.float64(0.45)]

h_range = np.arange(-5, 5, 0.1)
k_range = np.arange(-5, 5, 0.1)
l_cuts = [0, 0.25, 0.5]

input_positions = model_positions_to_ASE_positions(model_1_positions)
unit_cell_ASE = unit_cell(a, b, c, alpha, beta, gamma, formula, input_positions)

bulk = unit_cell_ASE.repeat(bulk_dimensions)
#view(bulk)

#original_structure = io.read('CsV3Sb5.cif')
#bulk_original = original_structure.repeat(bulk_dimensions)
#view(bulk_original)

plot_XRD_pattern(h_range, k_range, l_cuts, bulk, twin_angles, twin_populations)


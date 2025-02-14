import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from ase import Atoms
from ase import io
from ase.visualize import view
from ase.geometry import cellpar_to_cell

def model_positions_to_ASE_positions(model_positions):
    input_pos_list = []
    for i in model_positions.values():
        for j in i:
            input_pos_list.append(j)
    return input_pos_list

def unit_cell(cell_params, formula, input_positions):
    a, b, c, alpha, beta, gamma = cell_params
    cell_matrix = cellpar_to_cell([a, b, c, alpha, beta, gamma])
    atoms = Atoms(
        formula,
        positions=np.dot(input_positions, cell_matrix),
        cell=cell_matrix,
        pbc=True,
    )

    return atoms

def in_plane_bulk(model_positions, cell_params, formula, in_plane_repeats):
    """Create a single in-plane bulk (one layer) by repeating a model."""
    a, b, c, alpha, beta, gamma = cell_params
    cell_matrix = cellpar_to_cell([a, b, c, alpha, beta, gamma])
    positions = model_positions_to_ASE_positions(model_positions)
    unit_cell = Atoms(formula, positions=np.dot(positions, cell_matrix), cell=cell_matrix, pbc=True)

    bulk_layer = unit_cell.repeat((in_plane_repeats[0], in_plane_repeats[1], 1))
    
    return bulk_layer

def create_two_layer_unit_cell(layer_1, layer_2, interlayer_shift=None):
    """Creates a new unit cell with two stacked layers, applying an interlayer shift."""
    
    # Make copies to avoid modifying original layers
    layer_1 = layer_1.copy()
    layer_2 = layer_2.copy()

    # Apply interlayer shift to layer_2 if specified
    if interlayer_shift:
        dx, dy = interlayer_shift
        layer_2.positions[:, 0] += dx
        layer_2.positions[:, 1] += dy

    # Stack layer_2 on top of layer_1
    layer_2.positions[:, 2] += layer_1.cell[2, 2]  # Shift by c-axis value
    
    # Combine both layers into a single Atoms object
    two_layer_unit = layer_1 + layer_2  # ASE allows adding Atoms objects
    
    # Update the unit cell height to fit both layers
    new_cell = layer_1.cell.copy()
    new_cell[2, 2] *= 2  # Double the c-axis length
    two_layer_unit.set_cell(new_cell)
    
    return two_layer_unit

def build_full_bulk(two_layer_unit, num_repeats):
    """Creates the full bulk structure by repeating the two-layer unit cell in the z-direction."""
    return two_layer_unit.repeat((1, 1, num_repeats[2]))

def atomic_form_factor(atom, q):
    factors = {
        'Cs': lambda q: 1/(1+(q/50)**2)**2 ,    # Z = 55
        'V': lambda q: 1/(1+(q/23)**2)**2 , # Z = 23
        'Sb': lambda q:  1/(1+(q/46)**2)**2  # Z = 51
    }
    return factors[atom](q)

def F_hkl(atoms, G_vectors, q_values, batch_size_atoms=500, batch_size_hkl=5000):    
    """Computes the structure factor F_hkl in a memory-efficient way using batch processing."""
    
    atom_types = np.unique(atoms.get_chemical_symbols())
    form_factors = {atom: atomic_form_factor(atom, q_values) for atom in atom_types}

    positions = atoms.get_positions()
    symbols = np.array(atoms.get_chemical_symbols())

    num_hkl = G_vectors.shape[0]
    num_atoms = positions.shape[0]
    
    F_hkl_values = np.zeros(num_hkl, dtype=np.complex128)

    # Process hkl in batches
    for i in range(0, num_hkl, batch_size_hkl):
        G_batch = G_vectors[i:i + batch_size_hkl]  # Select batch of G vectors
        q_batch = q_values[i:i + batch_size_hkl]

        F_batch = np.zeros(G_batch.shape[0], dtype=np.complex128)

        # Process atoms in smaller batches
        for j in range(0, num_atoms, batch_size_atoms):
            pos_batch = positions[j:j + batch_size_atoms]  # Atom positions batch
            sym_batch = symbols[j:j + batch_size_atoms]  # Atom types batch
            
            # Compute form factors for this batch of atoms
            f_atoms = np.array([form_factors[s][i:i + batch_size_hkl] for s in sym_batch])  # Shape (batch_size_atoms, batch_size_hkl)

            # Compute phase terms for this subset of atoms
            phase_terms = np.exp(-1j * 2 * np.pi * np.dot(pos_batch, G_batch.T))  # Shape (batch_size_atoms, batch_size_hkl)

            # Correct broadcasting for multiplication
            F_batch += np.sum(f_atoms * phase_terms, axis=0)  # Sum over atoms in batch

        # Store the computed F_hkl values for this batch of hkl
        F_hkl_values[i:i + batch_size_hkl] = F_batch  

    return F_hkl_values

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

def rotate_reciprocal_lattice(hkl_array, angle_deg, rotation_matrices):
    h_k_rot = np.dot(hkl_array[:, :2], rotation_matrices[angle_deg].T)  # Vectorized rotation
    return np.hstack((h_k_rot, hkl_array[:, 2][:, np.newaxis]))  # Preserve shape

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

def DW_correction(q, B = 0.5):
    return np.float64(np.exp(-B * q**2))

def I_hkl(atoms, hkl_array, r_lattice, twin_angles, twin_fractions, rotation_matrices, wavelength=1.54, batch_size_hkl=5000):
    num_hkl = hkl_array.shape[0]
    F_total = np.zeros(num_hkl, dtype=np.complex128)

    for twin_angle, twin_fraction in zip(twin_angles, twin_fractions):
        hkl_twin = rotate_reciprocal_lattice(hkl_array, twin_angle, rotation_matrices)
        G_vectors = np.dot(hkl_twin, r_lattice.T)
        q_values = np.linalg.norm(G_vectors, axis=1)

        for i in range(0, num_hkl, batch_size_hkl):
            G_batch = G_vectors[i:i + batch_size_hkl]
            q_batch = q_values[i:i + batch_size_hkl]

            F_twin = (
                F_hkl(atoms, G_batch, q_batch) * twin_fraction *
                DW_correction(q_batch) * LP_correction(q_batch, wavelength)
            )
            F_total[i:i + batch_size_hkl] += F_twin

    return np.abs(F_total)**2

def plot_XRD_pattern(h_range, k_range, l_cuts, atoms, twin_angles, twin_fractions):
    recip_lattice = np.round(np.linalg.inv(atoms.get_cell().T) * (2 * np.pi), decimals=10)
    rotation_matrices = precompute_rotation_matrices(twin_angles)
    
    h_grid, k_grid = np.meshgrid(h_range, k_range, indexing='ij')  # Ensure correct indexing
    hkl_array = np.column_stack((h_grid.ravel(), k_grid.ravel()))  # Flatten h and k values

    for l in l_cuts:
        hkl_values = np.hstack((hkl_array, np.full((hkl_array.shape[0], 1), l)))  # Add l column
        intensity = I_hkl(atoms, hkl_values, recip_lattice, twin_angles, twin_fractions, rotation_matrices)
        intensity = intensity.reshape(len(h_range), len(k_range))  # Reshape back to mesh

        intensity /= (np.max(intensity)/10e3)  # Normalize; the CDw peaks intensities are 10^4 to 10^6 times smaller than the Braggs
        intensity = gaussian_filter(intensity, sigma=0.5)  # Smooth
        intensity = np.log1p(intensity)  # Log scaling

        # Plot the intensity map
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111)

        # Plot the intensity as a heatmap
        c = ax.pcolormesh(h_grid, k_grid, intensity, shading='auto', cmap='viridis')
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
cell_params = (a, b, c, alpha, beta, gamma)
formula = "Cs2V4Sb6"
atom_types = ['Cs', 'V', 'Sb']  # Atom types for each position
bulk_dimensions = (10, 10, 15)
twin_angles, twin_populations = [0, 120, 240], [np.float64(0.33), np.float64(0.33), np.float64(0.33)]

h_range = np.arange(-4, 4, 0.1)
k_range = np.arange(-4, 4, 0.1)
l_cuts = [np.float64(0), np.float64(0.25), np.float64(0.5)]

#input_positions = model_positions_to_ASE_positions(model_2_positions)
#unit_cell_ASE = unit_cell(cell_params, formula, input_positions)

#bulk = unit_cell_ASE.repeat(bulk_dimensions)
#view(bulk)

#original_structure = io.read('CsV3Sb5.cif')
#bulk_original = original_structure.repeat(bulk_dimensions)
#view(bulk_original)

layer_1 = in_plane_bulk(model_1_positions, cell_params, formula, bulk_dimensions)
layer_2 = in_plane_bulk(model_2_positions, cell_params, formula, bulk_dimensions)

interlayer_shift = (0.5 * a, 0)
two_layer_cell = create_two_layer_unit_cell(layer_1, layer_2, interlayer_shift)

bulk = build_full_bulk(two_layer_cell, bulk_dimensions)

plot_XRD_pattern(h_range, k_range, l_cuts, bulk, twin_angles, twin_populations)


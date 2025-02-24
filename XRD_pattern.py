import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from ase import Atoms
from ase import io
from ase.visualize import view
from ase.geometry import cellpar_to_cell
from multiprocessing import Pool

### Unit cell creation and modification 

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

def create_2x2x2_unit_cell(layer_1, layer_2, interlayer_shift=None):
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

def create_2x2x4_unit_cell(layer_1, layer_2, layer_3, layer_4, interlayer_shifts=None):
    """Creates a new unit cell with four stacked layers, applying interlayer shifts if specified.
    
    Parameters:
    - layer_1, layer_2, layer_3, layer_4: ASE Atoms objects for the four layers
    - interlayer_shifts: List of (dx, dy) shifts for layers 2, 3, and 4 relative to the previous one
    """

    # Make copies to avoid modifying the original layers
    layers = [layer_1.copy(), layer_2.copy(), layer_3.copy(), layer_4.copy()]

    # Default shifts if not provided
    if interlayer_shifts is None:
        interlayer_shifts = [(0, 0), (0, 0), (0, 0)]  # No shift by default

    if len(interlayer_shifts) != 3:
        raise ValueError("interlayer_shifts must be a list of 3 (dx, dy) tuples.")

    # Apply interlayer shifts to layers 2, 3, and 4
    for i, shift in enumerate(interlayer_shifts):
        dx, dy = shift
        layers[i + 1].positions[:, 0] += dx  # Shift in x
        layers[i + 1].positions[:, 1] += dy  # Shift in y

    # Correctly stack layers along the c-axis
    for i in range(1, 4):  # Start from layer 2, shifting based on the previous one
        layers[i].positions[:, 2] += i * layers[0].cell[2, 2]  # Ensure proper stacking

    # Combine all four layers into a single Atoms object
    four_layer_unit = sum(layers, Atoms())

    # Update the unit cell height to fit four layers
    new_cell = layers[0].cell.copy()
    new_cell[2, 2] *= 4  # Correctly increase the c-axis length
    four_layer_unit.set_cell(new_cell)

    return four_layer_unit

def build_full_bulk(unit_cell, num_repeats):
    """Creates the full bulk structure by repeating the two-layer unit cell in the z-direction."""
    return unit_cell.repeat((1, 1, num_repeats[2]))

def degrees_to_radians(angle_deg):
    return np.radians(angle_deg)

def radians_to_degrees(angle_rad):
    return np.degrees(angle_rad)

def unit_cell_to_metric_tensor(cell_params):
    """Convert unit cell parameters to metric tensor G."""
    a, b, c, alpha, beta, gamma = cell_params
    alpha, beta, gamma = map(degrees_to_radians, [alpha, beta, gamma])

    G = np.array([
        [a**2, a*b*np.cos(gamma), a*c*np.cos(beta)],
        [a*b*np.cos(gamma), b**2, b*c*np.cos(alpha)],
        [a*c*np.cos(beta), b*c*np.cos(alpha), c**2]
    ])
    return G

def metric_tensor_to_unit_cell(G):
    """Convert metric tensor G back to unit cell parameters."""
    a_new = np.sqrt(G[0, 0])
    b_new = np.sqrt(G[1, 1])
    c_new = np.sqrt(G[2, 2])

    alpha_new = radians_to_degrees(np.arccos(G[1, 2] / (b_new * c_new)))
    beta_new = radians_to_degrees(np.arccos(G[0, 2] / (a_new * c_new)))
    gamma_new = radians_to_degrees(np.arccos(G[0, 1] / (a_new * b_new)))

    return a_new, b_new, c_new, alpha_new, beta_new, gamma_new

def apply_strain(cell_params, strain_tensors, twin_angles, rotation_matrices):
    """Applies different rotated strain matrices to the unit cell for each twin variant efficiently."""

    # Ensure strain_tensors is (N_twin, 3, 3)
    strain_tensors = np.array(strain_tensors)
    if strain_tensors.ndim == 2:  # If shape is (3, 3), expand to (N_twin, 3, 3)
        strain_tensors = np.tile(strain_tensors, (len(twin_angles), 1, 1))
    
    # Convert input unit cell parameters to metric tensor
    G = unit_cell_to_metric_tensor(cell_params)
    
    # Stack rotation matrices for all twin angles (N_twin, 3, 3)
    R_matrices = np.array([rotation_matrices[angle] for angle in twin_angles])

    # Rotate the strain tensors: ε' = R ε R^T
    rotated_strains = np.einsum('nij,njk,nlk->nil', R_matrices, strain_tensors, R_matrices)

    # Apply the strain to G: G' = (I + ε)^T G (I + ε)
    I_plus_strains = np.identity(3) + rotated_strains
    G_new = np.einsum('nij,jk,nlk->nil', I_plus_strains, G, I_plus_strains)

    # Convert back to unit cell parameters for each twin variant
    strain_unit_cell_params = np.array([metric_tensor_to_unit_cell(G_new[i]) for i in range(len(twin_angles))])

    return strain_unit_cell_params  # Returns a NumPy array of shape (N_twin, 6)

def generate_bulk(cell_params, formula, bulk_dimensions, model1, model2 = None):
    """Helper function to generate a bulk structure for a given strained unit cell."""
    layer_1 = in_plane_bulk(model1, cell_params, formula, bulk_dimensions)
    layer_2 = in_plane_bulk(model2, cell_params, formula, bulk_dimensions)

    interlayer_shift = (0.0 * cell_params[0], 0.5 * cell_params[1])  # Adjust shift based on a
    #interlayer_shift = [interlayer_shift]*3
    two_layer_cell = create_2x2x2_unit_cell(layer_1, layer_2, interlayer_shift)
    #four_layer_cell = create_2x2x4_unit_cell(layer_1, layer_2, layer_2, layer_2, interlayer_shift)
    full_bulk = build_full_bulk(two_layer_cell, bulk_dimensions)

    return full_bulk

def compute_twin_bulks_parallel(strained_unit_cells, formula, bulk_dimensions, model1, model2=None):
    """Computes twin bulk structures in parallel, ensuring output remains a list of ASE Atoms objects."""
    
    # Create argument tuples for each bulk
    args = [(cell_params, formula, bulk_dimensions, model1, model2) for cell_params in strained_unit_cells]

    with Pool() as pool:
        twin_bulks = pool.starmap(generate_bulk, args)  # Use starmap to unpack arguments

    return twin_bulks  # Return as a list, NOT a NumPy array


### Diffraction pattern calculation functions

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
    """Precompute full 3D rotation matrices for each twin angle."""
    angles_rad = np.radians(angles_deg)
    cos_vals = np.cos(angles_rad)
    sin_vals = np.sin(angles_rad)

    # Create full 3D rotation matrices (assuming rotation about z-axis)
    return {
        angle: np.array([
            [c, -s, 0],
            [s,  c, 0],
            [0,  0, 1]  # No rotation in the z-direction
        ])
        for angle, c, s in zip(angles_deg, cos_vals, sin_vals)
    }
def rotate_reciprocal_lattice(hkl_array, angle_deg, rotation_matrices):
    h_k_rot = np.dot(hkl_array[:, :2], rotation_matrices[angle_deg][:2, :2].T)  # Vectorized rotation
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

def compute_F_hkl_batch(args):
    """Wrapper function to compute F_hkl for a batch."""
    twin_bulk, G_batch, q_batch, twin_fraction, wavelength = args
    return (
        F_hkl(twin_bulk, G_batch, q_batch) * twin_fraction #*
        #DW_correction(q_batch) * LP_correction(q_batch, wavelength)
    )

def I_hkl(twin_bulks, hkl_array, r_lattice, twin_angles, twin_fractions, rotation_matrices, wavelength=1.54, batch_size_hkl=20000):
    """Computes XRD intensities efficiently using batch processing."""
    
    num_hkl = hkl_array.shape[0]
    F_total = np.zeros(num_hkl, dtype=np.complex128)  # Accumulate structure factors

    for twin_idx, (twin_bulk, twin_angle, twin_fraction) in enumerate(zip(twin_bulks, twin_angles, twin_fractions)):
        # Rotate reciprocal lattice (uses precomputed matrices, already handled correctly)
        hkl_twin = rotate_reciprocal_lattice(hkl_array, twin_angle, rotation_matrices)
        G_vectors = np.dot(hkl_twin, r_lattice.T)  # Compute scattering vectors
        q_values = np.linalg.norm(G_vectors, axis=1)  # Compute |G|

        # Prepare batch arguments
        batch_args = [
            (twin_bulk, G_vectors[i:i + batch_size_hkl], q_values[i:i + batch_size_hkl], twin_fraction, wavelength)
            for i in range(0, num_hkl, batch_size_hkl)
        ]

        # Compute batches in parallel
        with Pool() as pool:
            F_batches = pool.map(compute_F_hkl_batch, batch_args)

        # Accumulate results
        for i, F_batch in enumerate(F_batches):
            start_idx = i * batch_size_hkl
            end_idx = min(start_idx + batch_size_hkl, num_hkl)
            np.add.at(F_total, np.arange(start_idx, end_idx), F_batch)

    return np.abs(F_total)**2  # Return intensity (|F|^2)

### Plotting and precomputing functions and values

def plot_XRD_pattern(h_range, k_range, l_cuts, twin_angles, twin_fractions, cell_params, strain_tensor = None):
    rotation_matrices = precompute_rotation_matrices(twin_angles)
    
    # Compute strained unit cells (vectorized)
    strained_unit_cells = apply_strain(cell_params, strain_tensor, twin_angles, rotation_matrices)

    # Compute twin bulks in parallel
    twin_bulks = compute_twin_bulks_parallel(strained_unit_cells, formula, bulk_dimensions, model_1_positions, model_2_positions)

    # Compute reciprocal lattice (using the first bulk, as they share the basis)
    recip_lattice = np.round(np.linalg.inv(twin_bulks[0].get_cell().T) * (2 * np.pi), decimals=10)

    h_grid, k_grid = np.meshgrid(h_range, k_range, indexing='ij')  # Ensure correct indexing
    hkl_array = np.column_stack((h_grid.ravel(), k_grid.ravel()))  # Flatten h and k values

    for l in l_cuts:
        hkl_values = np.hstack((hkl_array, np.full((hkl_array.shape[0], 1), l)))  # Add l column
        intensity = I_hkl(twin_bulks, hkl_values, recip_lattice, twin_angles, twin_fractions, rotation_matrices)
        intensity = intensity.reshape(len(h_range), len(k_range))  # Reshape back to mesh

        intensity /= (np.max(intensity)/10e1)  # Normalize; the CDw peaks intensities are 10^4 to 10^6 times smaller than the Braggs
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

### Models with atoms positions

# Tri-Hexagonal
model_1_positions = {
    "Cs": [(0, 0, 0.750748), 
           (0, 0, 0.249252), 
           (0.25, 0.25, 0.25), 
           (0.75, 0.75, 0.75)],
    "V": [
        (0.3785912, 0.126306, 0),
        (0.6214088, 0.873694, 0),
        (0.6214088, 0.126306, 0),
        (0.3785912, 0.873694, 0),
        (0.24752, 0, 0),
        (0.75248, 0, 0),
        (0, 0.7474110, 0),
        (0, 0.252589, 0),
    ],
    "Sb": [
        (0, 0.332945, 0.6215610),
        (0, 0.667055, 0.378439),
        (0, 0.332945, 0.378439),
        (0, 0.667055, 0.6215610),
        (0.250563, 0.083484, 0.121629),
        (0.749437, 0.916516, 0.878371),
        (0.250563, 0.916516, 0.878371),
        (0.749437, 0.083484, 0.121629), 
        (0.749437, 0.916516, 0.121629),
        (0.250563, 0.083484, 0.878371),
        (0.749437, 0.083484, 0.878371), 
        (0.250563, 0.916516, 0.121629),
        (0, 0, 0.5),
        (0.25, 0.25, 0),
        (0.75, 0.25, 0),
        (0, 0, 0),
    ],
}

# SOD
model_2_positions = {
    "Cs": [(0, 0, 0.749239), 
           (0, 0, 0.250761),
           (0.25, 0.25, 0.25),
           (0.75, 0.75, 0.75),
           ],
    "V": [
        (0.3715413, 0.123737, 0),
        (0.6284587, 0.876263, 0),
        (0.6284587, 0.123737, 0),
        (0.3715413, 0.876263, 0),
        (0.25252, 0, 0),
        (0.74748, 0, 0),
        (0, 0.7525912, 0),
        (0, 0.2474088, 0),
    ],
    "Sb": [
        (0, 0.333695, 0.6225110),
        (0, 0.666305, 0.377489),
        (0, 0.333695, 0.377489),
        (0, 0.666305, 0.6225110),
        (0.249444, 0.083175, 0.122479), 
        (0.750556, 0.916825, 0.877521), 
        (0.750556, 0.916825, 0.122479),
        (0.249444, 0.083175, 0.877521), 
        (0.750556, 0.083175, 0.877521),
        (0.249444, 0.916825, 0.122479),
        (0.249444, 0.916825, 0.877521),
        (0.750556, 0.083175, 0.122479), 
        (0, 0, 0.5),
        (0.25, 0.25, 0),
        (0.75, 0.25, 0),
        (0, 0, 0),
    ],
}

## Parameters for input

# Initial undistorted lattice constants (Å)
a, b, c = 10.971314, 18.9833, 18.51410  
alpha, beta, gamma = 90, 90, 90
cell_params = (a, b, c, alpha, beta, gamma)

# Applying strain

# Define a small strain tensor
strain_tensor = np.array([
    [0.0, 0.0, 0.0],  # ε_xx, ε_xy, ε_xz
    [0.0, 0.0, 0.00], # ε_yx, ε_yy, ε_yz
    [0.0, 0.00, 0.0]       # ε_xz, ε_yz, ε_zz
])

################################
formula = "Cs4V8Sb16"
atom_types = ['Cs', 'V', 'Sb']  # Atom types for each position
bulk_dimensions = (10, 10, 10)
twin_angles, twin_populations = [0, 120, 240], [np.float64(0.0), np.float64(0.33), np.float64(0.33)]

h_range = np.arange(-3, 3, 0.1)
k_range = np.arange(-3, 3, 0.1)

l_cuts = [np.float64(0), np.float64(0.25), np.float64(0.50)]

#input_positions = model_positions_to_ASE_positions(model_2_positions)
#unit_cell_ASE = unit_cell(cell_params, formula, input_positions)

#bulk = unit_cell_ASE.repeat(bulk_dimensions)
#view(bulk)

#original_structure = io.read('CsV3Sb5.cif')
#bulk_original = original_structure.repeat(bulk_dimensions)
#view(bulk_original)

plot_XRD_pattern(h_range, k_range, l_cuts, twin_angles, twin_populations, cell_params, strain_tensor)


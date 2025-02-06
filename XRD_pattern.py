import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from ase import Atoms
from ase import io
from ase.visualize import view
from mpl_toolkits.mplot3d import Axes3D
import re

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
    
    atoms = Atoms(
        formula,
        positions=input_positions,
        cell=[a, b, c, alpha, beta, gamma],
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

def rotate_reciprocal_lattice(hkl, angle_deg):
    h, k, l = hkl
    angle_rad = np.radians(angle_deg)
    h_rot = h * np.cos(angle_rad) - k * np.sin(angle_rad)
    k_rot = h * np.sin(angle_rad) + k * np.cos(angle_rad)
    return (h_rot, k_rot, l)

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

def DW_correction(q, B = 0.005):
    return np.float64(np.exp(-B * q**2))

def I_hkl(atoms, hkl, r_lattice, twin_angles, twin_fractions, wavelength = 1.54):
    I_total = 0
    for twin_angle, twin_fraction in zip(twin_angles, twin_fractions):
        hkl_twin = rotate_reciprocal_lattice(hkl, twin_angle)
        G = np.dot(hkl_twin, r_lattice)
        q = np.linalg.norm(G)  # Magnitude of reciprocal lattice vector
        F_twin = F_hkl(atoms, G, q) * twin_fraction * DW_correction(q) * LP_correction(q, wavelength)
        I_total += np.abs(F_twin)**2
    return I_total

def plot_XRD_pattern(h_range, k_range, l_cuts, atoms, twin_angles, twin_fractions):
    recip_lattice = np.round(np.linalg.inv(atoms.get_cell().T) * (2 * np.pi), decimals=10)
    #h_range_int, k_range_int = np.unique(np.floor(h_range).astype(int)), np.unique(np.floor(k_range).astype(int))
    for l in l_cuts:
        h, k = np.meshgrid(h_range, k_range)
        intensity = np.zeros((len(h_range), len(k_range)))

        for i in range(len(h_range)):
            for j in range(len(k_range)):
                hkl = (h[i, j], k[i, j], l)
                hkl = tuple(np.round(hkl, decimals=10))
                intensity[i, j] = I_hkl(atoms, hkl, recip_lattice, twin_angles, twin_fractions)

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

class CustomAtoms:
    def __init__(self, formula, positions, cell, pbc=True):
        self.formula = formula
        self.positions = np.array(positions)  # Atomic positions
        self.pbc = pbc  # Periodic Boundary Conditions flag

        # If cell is given as 3 parameters, convert it to 6
        if len(cell) == 3:
            # Assuming cell is passed as lattice vectors
            self.cell = self._calculate_cell_from_vectors(cell)
        elif len(cell) == 6:
            # Assuming cell is passed as lattice parameters [a, b, c, alpha, beta, gamma]
            self.cell = self._calculate_cell_matrix(cell)
        else:
            raise ValueError("The cell must be specified with either 3 lattice vectors or 6 parameters (a, b, c, alpha, beta, gamma).")

        # Ensure the cell is 3x3
        if self.cell.ndim != 2 or self.cell.shape != (3, 3):
            raise ValueError(f"The cell must be a 3x3 matrix, but received shape {self.cell.shape}.")

        # Parse formula to determine the atom types and their counts
        self.atom_types = self.parse_formula(formula)
        
        # Ensure number of atom types matches number of positions
        total_atoms = len(self.atom_types)
        if total_atoms != len(self.positions):
            raise ValueError(f"The number of atom types ({total_atoms}) does not match the number of positions ({len(self.positions)}).")

    def _calculate_cell_matrix(self, cell):
        """
        Convert cell parameters [a, b, c, alpha, beta, gamma] to a 3x3 matrix.
        Angles are given in degrees, so we convert them to radians.
        """
        a, b, c, alpha, beta, gamma = cell
        alpha = np.radians(alpha)
        beta = np.radians(beta)
        gamma = np.radians(gamma)

        # Calculate the 3x3 matrix using the formula
        cell_matrix = np.zeros((3, 3))
        cell_matrix[0] = [a, 0, 0]
        cell_matrix[1] = [b * np.cos(gamma), b * np.sin(gamma), 0]
        cell_matrix[2] = [
            c * np.cos(beta),
            c * (np.cos(alpha) - np.cos(beta) * np.cos(gamma)) / np.sin(gamma),
            c * np.sqrt(1 - np.cos(beta)**2 - (np.cos(alpha) - np.cos(beta) * np.cos(gamma))**2)
        ]
        return cell_matrix

    def _calculate_cell_from_vectors(self, cell_vectors):
        """
        Converts 3 lattice vectors into a 3x3 matrix (direct lattice matrix).
        """
        if len(cell_vectors) != 3:
            raise ValueError("Cell vectors must be a list of 3 vectors.")
        
        return np.array(cell_vectors)

    def get_cell(self):
        return self.cell

    def get_positions(self):
        return self.positions

    def get_chemical_symbols(self):
        return self.atom_types
    
    def parse_formula(self, formula):
        """
        Parses the formula string to determine atom types and counts.
        Handles both single-letter and multi-letter atomic symbols.
        Example:
        For formula 'CsV3Sb5', it should return ['Cs', 'V', 'V', 'V', 'Sb', 'Sb', 'Sb', 'Sb', 'Sb']
        """
        # Regular expression to match elements and their counts
        pattern = r'([A-Z][a-z]?)(\d*)'
        atom_counts = []
        atom_types = []
        
        for element, count in re.findall(pattern, formula):
            count = int(count) if count else 1  # Default count to 1 if not specified
            atom_types.extend([element] * count)
        
        return atom_types

    def replicate(self, repeats):
        """
        Replicate the unit cell to create a bulk structure.
        `repeats` is a tuple (nx, ny, nz) indicating the number of times to replicate along each axis.
        """
        if len(repeats) != 3:
            raise ValueError("The input 'repeats' must be a tuple with three elements (nx, ny, nz).")
        
        nx, ny, nz = repeats
        replicated_positions = []
        
        # Replicate positions in x, y, and z directions
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    # Translate the positions by the corresponding multiples of the unit cell
                    offset = np.array([i, j, k])
                    
                    # Add the positions with the offset
                    for pos in self.positions:
                        replicated_positions.append(pos + offset)
        
        # Replicate the atom types the same number of times as the number of positions
        total_atoms = len(self.atom_types)
        replicated_atom_types = self.atom_types * (nx * ny * nz)

        # Ensure the number of atom types matches the number of positions
        if len(replicated_atom_types) != len(replicated_positions):
            print(f"Error! Number of atom types: {len(replicated_atom_types)}, number of positions: {len(replicated_positions)}")
            raise ValueError(f"The number of atom types ({len(replicated_atom_types)}) does not match the number of positions ({len(replicated_positions)}).")
        
        # Return a new CustomAtoms object with the replicated positions and atom types
        return CustomAtoms(self.formula, replicated_positions, self.cell, self.pbc)







    
    def visualize_3d(self):
        """Visualize the atomic positions in 3D with different colors for each atom type."""
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        
        # Extract x, y, z coordinates from positions
        x, y, z = self.positions[:, 0], self.positions[:, 1], self.positions[:, 2]
        
        # Map atom types to colors
        atom_colors = {'Cs': 'red', 'V': 'blue', 'Sb': 'green', 'X': 'black'}  # Example color map
        colors = [atom_colors.get(atom, 'gray') for atom in self.atom_types]
        
        # Scatter plot of atomic positions
        ax.scatter(x, y, z, c=colors, marker='o')
        
        # Labels and titles
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('3D Atomic Structure')
        
        # Show plot
        plt.show()


    def __repr__(self):
        return f"CustomAtoms(formula={self.formula}, cell={self.cell}, positions={self.positions})"

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
bulk_dimensions = (5, 5, 5)
twin_angles, twin_populations = [0, 120, 240], [np.float64(1), np.float64(0.0), np.float64(0.0)]

h_range = np.arange(-5, 5, 0.1)
k_range = np.arange(-5, 5, 0.1)
l_cuts = [0, 0.25, 0.5]

input_positions = model_positions_to_ASE_positions(model_1_positions)
unit_cell_ASE = unit_cell(a, b, c, alpha, beta, gamma, formula, input_positions)

bulk = unit_cell_ASE.repeat(bulk_dimensions)

#view(bulk)

original_structure = io.read('CsV3Sb5.cif')
bulk_original = original_structure.repeat(bulk_dimensions)
#view(bulk_original)

plot_XRD_pattern(h_range, k_range, l_cuts, bulk_original, twin_angles, twin_populations)


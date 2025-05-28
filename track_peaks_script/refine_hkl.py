import numpy as np
import pandas as pd
import argparse
from scipy.optimize import least_squares

# -- Metric tensor constructors for each symmetry --
def triclinic_G(p):
    G11, G22, G33, G12, G13, G23 = p
    return np.array([[G11, G12, G13],
                     [G12, G22, G23],
                     [G13, G23, G33]])

def monoclinic_G(p):
    G11, G22, G33, G13 = p
    return np.array([[G11, 0,    G13],
                     [0,   G22, 0  ],
                     [G13, 0,    G33]])

def orthorhombic_G(p):
    G11, G22, G33 = p
    return np.diag([G11, G22, G33])

def tetragonal_G(p):
    G11, G33 = p
    return np.diag([G11, G11, G33])

def rhombohedral_G(p):
    G11, G12 = p
    return np.array([[G11, G12, G12],
                     [G12, G11, G12],
                     [G12, G12, G11]])

def hexagonal_G(p):
    G11, G33 = p
    G12 = -G11 / 2
    return np.array([[G11, G12, 0],
                     [G12, G11, 0],
                     [0,   0,   G33]])

def cubic_G(p):
    G11, = p
    return np.diag([G11, G11, G11])

# Symmetry map: name -> (number of params, G-function)
symmetries = {
    'triclinic':    (6, triclinic_G),
    'monoclinic':   (4, monoclinic_G),
    'orthorhombic': (3, orthorhombic_G),
    'tetragonal':   (2, tetragonal_G),
    'rhombohedral': (2, rhombohedral_G),
    'hexagonal':    (2, hexagonal_G),
    'cubic':        (1, cubic_G)
}

# Compute d-spacing from G and hkl vectors
def compute_d(G, hkl):
    # hkl: (N,3)
    vals = np.dot(hkl, G)   # (N,3)
    hGh = np.sum(vals * hkl, axis=1)
    return 1.0 / np.sqrt(hGh)

# Residuals for least_squares, with optional intensity weighting
def residuals(p, hkl, d_obs, sigma, weight, G_func):
    G = G_func(p)
    d_calc = compute_d(G, hkl)
    res = (d_calc - d_obs) / sigma
    return res * weight

# R-value
def R_value(d_obs, d_calc):
    return np.sum(np.abs(d_obs - d_calc)) / np.sum(d_obs)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Index CDW unit cell from d_hkl values')
    parser.add_argument('input_csv', help='Input CSV with columns h,k,l,d,err,I')
    parser.add_argument('output_csv', help='Output CSV with ranking and parameters')
    parser.add_argument('--use-intensity', action='store_true',
                        help='Weight residuals by normalized intensity')
    args = parser.parse_args()

    df = pd.read_csv(args.input_csv)
    # allow fractional h,k,l for CDW peaks
    default = False
    hkl = df[['h','k','l']].astype(float).values
    d_obs = df['d'].values
    sigma = df['err'].values
    I = df['I'].values
    # weight: normalized sqrt intensity or uniform
    if args.use_intensity:
        # normalize to mean=1
        weight = np.sqrt(I / np.mean(I))
    else:
        weight = np.ones_like(d_obs)

    results = []
    for name, (npar, G_func) in symmetries.items():
        # initial guess: based on average
        mean_d = np.mean(d_obs)
        g0 = 1.0 / mean_d**2
        p0 = np.ones(npar) * g0
        sol = least_squares(residuals, p0,
                            args=(hkl, d_obs, sigma, weight, G_func),
                            xtol=1e-12, ftol=1e-12)
        p_fit = sol.x
        G_fit = G_func(p_fit)
        d_calc = compute_d(G_fit, hkl)
        chi2 = np.sum(((d_calc - d_obs)/sigma)**2)
        dof = len(d_obs) - npar
        chi2_red = chi2 / dof if dof > 0 else np.nan
        Rval = R_value(d_obs, d_calc)
        AIC = chi2 + 2 * npar
        results.append({
            'symmetry': name,
            'n_params': npar,
            'chi2_red': chi2_red,
            'R': Rval,
            'AIC': AIC,
            **{f'p{i+1}': val for i, val in enumerate(p_fit)}
        })

    out_df = pd.DataFrame(results)
    out_df = out_df.sort_values('AIC').reset_index(drop=True)
    out_df.to_csv(args.output_csv, index=False)
    print(f'Results written to {args.output_csv}')

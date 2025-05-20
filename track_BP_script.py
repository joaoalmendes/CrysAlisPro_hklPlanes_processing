import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

X_a_0, X_b_0, Y_c_0 = 741, 846, 463       # reference positions, taken from (80K, 0.0V)

# ─── Calibration constants (Å^-1 per pixel) ─────────────────────────────────
F_HK = 0.00813008   # a/h & b/k directions
F_LC = 0.01578947   # c/l direction

# ─── Filepaths ───────────────────────────────────────────────────────────────
STEADY_FILE = 'track_BP_data.txt'
ALL_FILE    = 'track_BP_data_full.txt'

# ─── Columns in your data files ──────────────────────────────────────────────
COLS = ['T_K', 'V_V', 'strain', 'X_a', 'X_b', 'Y_c']

# ─── Load data, skipping comment lines ────────────────────────────────────────
def load_data(path):
    return pd.read_csv(path, sep=r'\s+', comment='#', names=COLS)

# ─── Compute Δq and θ with per‐temperature reference ──────────────────────────
def compute_dq_theta(df):
    # 1) find reference (min‐voltage) row per T_K
    refs = (
        df.loc[df.groupby('T_K')['V_V'].idxmin()]
          .set_index('T_K')[['X_a','X_b','Y_c']]
          .rename(columns=lambda c: f'{c}_ref')
    )
    # 2) join back onto full df
    df = df.join(refs, on='T_K')
    # 3) pixel shifts
    df['dX_a'] = df['X_a'] - X_a_0
    df['dX_b'] = df['X_b'] - X_b_0
    df['dY_c'] = df['Y_c'] - Y_c_0
    # 4) q‐space shifts
    df['dq_a'] = df['dX_a'] * F_HK
    df['dq_b'] = df['dX_b'] * F_HK
    df['dq_c'] = df['dY_c'] * F_LC
    # 5) magnitude and angle
    df['dq']    = np.sqrt(df.dq_a**2 + df.dq_b**2 + df.dq_c**2)
    df['theta'] = np.degrees(np.arctan2(df.dq_c, df.dq_b))
    return df

# ─── Simple linear fit helper ────────────────────────────────────────────────
def fit_linear(x, y):
    slope, intercept, r, p, stderr = stats.linregress(x, y)
    return slope, intercept, stderr

# ─── Plot Δq vs. strain for a given subset ──────────────────────────────────
def plot_dq(df, out_png, title):
    slope, intercept, stderr = fit_linear(df['strain'], df['dq'])
    plt.figure()
    plt.scatter(df['strain'], df['dq'], label='data')
    xs = np.linspace(df['strain'].min(), df['strain'].max(), 100)
    plt.plot(xs, slope*xs + intercept, '--', label=f'slope={slope:.3e}±{stderr:.1e}')
    plt.xlabel('Strain (placeholder)')
    plt.ylabel(r'$|\Delta q|$ (Å$^{-1}$)')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    plt.close()


# ─── Plot θ vs. strain for given subset(s) ───────────────────────────────────
def plot_theta(subsets, out_png, title):
    plt.figure()
    for label, df in subsets.items():
        plt.scatter(df['strain'], df['theta'], label=label)
    plt.xlabel('Strain (placeholder)')
    plt.ylabel(r'θ (deg)')
    plt.title(title)
    plt.legend(ncol=2)
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    plt.close()


# ─── Main routine ───────────────────────────────────────────────────────────
if __name__ == '__main__':
    # Load
    df_steady = load_data(STEADY_FILE)
    df_all    = load_data(ALL_FILE)

    # Compute
    df_steady = compute_dq_theta(df_steady)
    df_all    = compute_dq_theta(df_all)

    # --- Δq vs. strain ---
    # 15K and 80K individually
    for T in [15, 80]:
        sub = df_steady[df_steady['T_K'] == T]
        plot_dq(sub, f'dq_vs_strain_{T}K.png', f'Δq vs. Strain @ {T} K')

    # Combined steady
    plot_dq(df_steady, 'dq_vs_strain_steady.png', 'Δq vs. Strain (steady, all phases)')

    # --- θ vs. strain ---
    # 15K & 80K only
    plot_theta(
        {'15 K': df_steady[df_steady['T_K'] == 15],
         '80 K': df_steady[df_steady['T_K'] == 80]},
        'theta_vs_strain_15_80K.png',
        'θ vs. Strain (15 K & 80 K)'
    )

    # All temperatures
    subsets = {f'{int(T)} K': df_all[df_all['T_K'] == T] for T in sorted(df_all['T_K'].unique())}
    plot_theta(subsets, 'theta_vs_strain_all.png', 'θ vs. Strain (all temperatures)')

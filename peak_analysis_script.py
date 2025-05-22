import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def compute_reciprocal_coordinates(df, r_hk, r_l):
    """
    Convert pixel positions (X_a, Y_b, Z_c) to reciprocal space coordinates (h, k, l)
    and compute |q| and theta for each point.
    
    Parameters:
    - df: DataFrame with columns 'X_a', 'Y_b', 'Z_c'
    - r_hk: Conversion ratio for hk-plane (pixels to reciprocal units)
    - r_l: Conversion ratio for l-direction (pixels to reciprocal units)
    
    Returns:
    - DataFrame with added columns 'h', 'k', 'l', 'q', 'theta'
    """
    # Compute k first, as h depends on k
    df['k'] = (df['Y_b'] * r_hk) / np.sin(np.pi / 3)
    # Compute h using the coupling term
    df['h'] = df['X_a'] * r_hk - df['k'] * np.cos(np.pi / 3)
    # Compute l
    df['l'] = df['Z_c'] * r_l
    # Compute magnitude |q|
    df['q'] = np.sqrt(df['h']**2 + df['k']**2 + df['l']**2)
    # Compute in-plane angle theta (in radians)
    df['theta'] = np.arctan2(df['k'], df['h'])
    return df

def compute_differences(df):
    """
    Compute differences (Δh, Δk, Δl, Δ|q|, Δθ) between consecutive points.
    
    Parameters:
    - df: DataFrame with columns 'h', 'k', 'l', 'q', 'theta'
    
    Returns:
    - DataFrame with columns 'pair_index', 'Delta_h', 'Delta_k', 'Delta_l', 'Delta_q', 'Delta_theta'
    """
    # Compute differences
    df_diff = df[['h', 'k', 'l', 'q', 'theta']].diff().iloc[1:].reset_index(drop=True)
    # Rename columns
    df_diff.columns = ['Delta_h', 'Delta_k', 'Delta_l', 'Delta_q', 'Delta_theta']
    # Add pair_index starting from 1
    df_diff['pair_index'] = range(1, len(df_diff) + 1)
    # Reorder columns
    return df_diff[['pair_index', 'Delta_h', 'Delta_k', 'Delta_l', 'Delta_q', 'Delta_theta']]

def plots(data, diff_data, T):
    # Load the processed data
    df_processed = pd.read_csv(data)
    df_differences = pd.read_csv(diff_data)

    # Create target directory
    orig_dir = os.getcwd()
    target_dir = os.path.join(orig_dir, 'BP_results', f'{T}_results')
    os.makedirs(target_dir, exist_ok=True)

    # Convert Delta_theta to degrees for the first set of plots
    df_differences['Delta_theta_deg'] = np.degrees(df_differences['Delta_theta'])

    # Plot 1: Delta_theta (degrees) vs pair_index
    plt.figure()
    plt.scatter(df_differences['pair_index'], df_differences['Delta_theta_deg'])
    plt.xlabel('Pair Index')
    plt.ylabel('Δθ (degrees)')
    plt.title('Change in θ between Consecutive Points')
    plt.savefig(os.path.join(target_dir, 'Delta_theta_vs_count.png'))
    plt.close()

    # Plot 2: Delta_q vs pair_index
    plt.figure()
    plt.scatter(df_differences['pair_index'], df_differences['Delta_q'])
    plt.xlabel('Pair Index')
    plt.ylabel('Δ|q|')
    plt.title('Change in |q| between Consecutive Points')
    plt.savefig(os.path.join(target_dir, 'Delta_q_vs_count.png'))
    plt.close()

    # Plot 3: Delta_h vs pair_index
    plt.figure()
    plt.scatter(df_differences['pair_index'], df_differences['Delta_h'])
    plt.xlabel('Pair Index')
    plt.ylabel('Δh')
    plt.title('Change in h between Consecutive Points')
    plt.savefig(os.path.join(target_dir, 'Delta_h_vs_count.png'))
    plt.close()

    # Plot 4: Delta_k vs pair_index
    plt.figure()
    plt.scatter(df_differences['pair_index'], df_differences['Delta_k'])
    plt.xlabel('Pair Index')
    plt.ylabel('Δk')
    plt.title('Change in k between Consecutive Points')
    plt.savefig(os.path.join(target_dir, 'Delta_k_vs_count.png'))
    plt.close()

    # Plot 5: Delta_l vs pair_index
    plt.figure()
    plt.scatter(df_differences['pair_index'], df_differences['Delta_l'])
    plt.xlabel('Pair Index')
    plt.ylabel('Δl')
    plt.title('Change in l between Consecutive Points')
    plt.savefig(os.path.join(target_dir, 'Delta_l_vs_count.png'))
    plt.close()

    # Convert theta to degrees for the second set of plots
    df_processed['theta_deg'] = np.degrees(df_processed['theta'])

    # Plot 6: theta (degrees) vs V_V
    plt.figure()
    plt.scatter(df_processed['V_V'], df_processed['theta_deg'])
    plt.xlabel('Voltage (V)')
    plt.ylabel('θ (degrees)')
    plt.title('θ vs Voltage')
    plt.savefig(os.path.join(target_dir, 'theta_vs_voltage.png'))
    plt.close()

    # Define V_V range for linear fit lines
    V_V_range = np.linspace(df_processed['V_V'].min(), df_processed['V_V'].max(), 100)

    # Plot 7: q vs V_V with linear fit
    m_q, c_q = np.polyfit(df_processed['V_V'], df_processed['q'], 1)
    q_fit = m_q * V_V_range + c_q
    plt.figure()
    plt.scatter(df_processed['V_V'], df_processed['q'], label='Data')
    plt.plot(V_V_range, q_fit, color='red', label=f'Fit: slope={m_q:.4f}')
    plt.xlabel('Voltage (V)')
    plt.ylabel('|q|')
    plt.title('|q| vs Voltage with Linear Fit')
    plt.legend()
    plt.savefig(os.path.join(target_dir, 'q_vs_voltage.png'))
    plt.close()

    # Plot 8: h vs V_V with linear fit
    m_h, c_h = np.polyfit(df_processed['V_V'], df_processed['h'], 1)
    h_fit = m_h * V_V_range + c_h
    plt.figure()
    plt.scatter(df_processed['V_V'], df_processed['h'], label='Data')
    plt.plot(V_V_range, h_fit, color='red', label=f'Fit: slope={m_h:.4f}')
    plt.xlabel('Voltage (V)')
    plt.ylabel('h')
    plt.title('h vs Voltage with Linear Fit')
    plt.legend()
    plt.savefig(os.path.join(target_dir, 'h_vs_voltage.png'))
    plt.close()

    # Plot 9: k vs V_V with linear fit
    m_k, c_k = np.polyfit(df_processed['V_V'], df_processed['k'], 1)
    k_fit = m_k * V_V_range + c_k
    plt.figure()
    plt.scatter(df_processed['V_V'], df_processed['k'], label='Data')
    plt.plot(V_V_range, k_fit, color='red', label=f'Fit: slope={m_k:.4f}')
    plt.xlabel('Voltage (V)')
    plt.ylabel('k')
    plt.title('k vs Voltage with Linear Fit')
    plt.legend()
    plt.savefig(os.path.join(target_dir, 'k_vs_voltage.png'))
    plt.close()

    # Plot 10: l vs V_V with linear fit
    m_l, c_l = np.polyfit(df_processed['V_V'], df_processed['l'], 1)
    l_fit = m_l * V_V_range + c_l
    plt.figure()
    plt.scatter(df_processed['V_V'], df_processed['l'], label='Data')
    plt.plot(V_V_range, l_fit, color='red', label=f'Fit: slope={m_l:.4f}')
    plt.xlabel('Voltage (V)')
    plt.ylabel('l')
    plt.title('l vs Voltage with Linear Fit')
    plt.legend()
    plt.savefig(os.path.join(target_dir, 'l_vs_voltage.png'))
    plt.close()

    return None

# Example usage
if __name__ == "__main__":
    # Read the data from the text file
    # Adjust the file path as needed
    T = "80K"
    df = pd.read_csv(T + '_BP_data.txt', sep='\s+' , comment='#', 
                     names=['T_K', 'V_V', 'strain', 'X_a', 'Y_b', 'Z_c'])

    # ─── Calibration constants (Å^-1 per pixel) ─────────────────────────────────
    r_hk = 0.00813008   # a/h & b/k directions
    r_l = 0.01578947   # c/l direction

    # Compute reciprocal coordinates, |q|, and theta
    df_processed = compute_reciprocal_coordinates(df.copy(), r_hk, r_l)

    # Compute differences between consecutive points
    df_differences = compute_differences(df_processed)

    # Save the processed data and differences to files
    df_processed.to_csv(f'{T}_BP_processed_data.csv', index=False)
    df_differences.to_csv(f'{T}_BP_differences.csv', index=False)

    plots(f'{T}_BP_processed_data.csv', f'{T}_BP_differences.csv', T)













#!/usr/bin/env python3
import ast
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def load_data(fn):
    df = pd.read_csv(
        fn,
        sep='\t',
        converters={
            'Reciprocal Space Plane': lambda s: s.strip(),
            'l (np.array)': ast.literal_eval,
            'Intensity (np.array)': ast.literal_eval,
        }
    )
    df['Temperature (K)'] = df['Temperature (K)'].astype(str).str.strip()
    df['Voltage (V)'] = df['Voltage (V)'].astype(float)
    return df

# Plot a single peak
def plot_curve(df, temp_id, voltage, plane, peak, rotate=False, save_fig=False, save_path='plot'):
    mask = (
        (df['Temperature (K)'] == temp_id) &
        (df['Voltage (V)'] == voltage) &
        (df['Reciprocal Space Plane'] == plane.strip())
    )
    df_sel = df[mask]
    if df_sel.empty:
        print("No matching data. Available options:\n")
        print(" Temperature IDs:", sorted(df['Temperature (K)'].unique()))
        print(" Voltages:", sorted(df['Voltage (V)'].unique()))
        print(" Planes:", sorted(df['Reciprocal Space Plane'].unique()))
        raise SystemExit(1)
    row = df_sel.iloc[0]
    l_vals = np.array(row['l (np.array)'], dtype=float)
    intens = np.array(row['Intensity (np.array)'], dtype=float)
    fig_size = (3,7) if rotate else (7,5)
    fig, ax = plt.subplots(figsize=fig_size)
    if rotate:
        ax.scatter(intens, l_vals, label=f'V={voltage}V')
        ax.set_xlabel('Intensity')
        ax.set_ylabel('l')
        ax.yaxis.set_label_position("right")
        ax.yaxis.tick_right()
    else:
        ax.scatter(l_vals, intens, label=f'V={voltage}V')
        ax.set_xlabel('l')
        ax.set_ylabel('Intensity')
    ax.legend(loc='best')
    ax.grid(True, axis='x', color='0.95')
    plt.tight_layout()
    if save_fig:
        fig.savefig(f"{save_path}_{temp_id}_V{voltage}_plane{plane}_peak{peak}.png", dpi=300)
    else:
        plt.show()
    plt.close(fig)

# General plotting for one peak across multiple voltages & planes
def plots_function(plot_dict, planes, temp_id, peak, voltages, mode, save_fig, save_path, rotate):
    colors = plt.cm.viridis(np.linspace(0,1,len(voltages)))
    for pl in planes:
        if mode == 'side_by_side':
            for idx, v in enumerate(voltages):
                if v not in plot_dict or pl not in plot_dict[v]: continue
                l_vals, intens = plot_dict[v][pl]
                fig_size = (3,7) if rotate else (7,5)
                fig, ax = plt.subplots(figsize=fig_size)
                if rotate:
                    ax.scatter(intens, l_vals, color=colors[idx], label=f'V={v}V')
                    ax.set_xlabel('Intensity')
                    ax.set_ylabel('l')
                    ax.yaxis.set_label_position("right")
                    ax.yaxis.tick_right()
                else:
                    ax.scatter(l_vals, intens, color=colors[idx], label=f'V={v}V')
                    ax.set_xlabel('l')
                    ax.set_ylabel('Intensity')
                ax.set_title(f'{temp_id}, Plane {pl}, Peak {peak}')
                ax.legend(loc='best')
                plt.tight_layout()
                if save_fig:
                    fig.savefig(f"{save_path}_{temp_id}_plane{pl}_V{v}_peak{peak}.png", dpi=300)
                plt.close(fig)

        elif mode == 'overlay':
            fig_size = (3,7) if rotate else (7,5)
            fig, ax = plt.subplots(figsize=fig_size)
            for idx, v in enumerate(voltages):
                if v not in plot_dict or pl not in plot_dict[v]: continue
                l_vals, intens = plot_dict[v][pl]
                if rotate:
                    ax.scatter(intens, l_vals, color=colors[idx], label=f'V={v}V')
                else:
                    ax.scatter(l_vals, intens, color=colors[idx], label=f'V={v}V')
            if rotate:
                ax.set_xlabel('Intensity')
                ax.set_ylabel('l')
                ax.yaxis.set_label_position("right")
                ax.yaxis.tick_right()
            else:
                ax.set_xlabel('l')
                ax.set_ylabel('Intensity')
            ax.set_title(f'{temp_id}, Plane {pl}, Peak {peak}')
            ax.legend(loc='best')
            plt.tight_layout()
            if save_fig:
                fig.savefig(f"{save_path}_{temp_id}_plane{pl}_overlay_peak{peak}.png", dpi=300)
            plt.close(fig)

        elif mode == 'stacked':
            fig_size = (3,7) if rotate else (7,5)
            fig, ax = plt.subplots(figsize=fig_size)
            shift = 0
            for idx, v in enumerate(voltages):
                if v not in plot_dict or pl not in plot_dict[v]: continue
                l_vals, intens = plot_dict[v][pl]
                max_I = max(intens)
                if rotate:
                    ax.scatter(intens+shift, l_vals, color='k')
                    ax.text(min(intens)+shift, max(l_vals), f'{v}V', va='center')
                else:
                    ax.scatter(l_vals, intens+shift, color='k')
                    ax.text(max(l_vals), min(intens)+shift, f'{v}V', va='center')
                shift += max_I*1.2
            if rotate:
                ax.set_xlabel('Intensity + shift')
                ax.set_ylabel('l')
                ax.yaxis.set_label_position("right")
                ax.yaxis.tick_right()
            else:
                ax.set_xlabel('l')
                ax.set_ylabel('Intensity + shift')
            ax.set_title(f'{temp_id}, Plane {pl}, Peak {peak}')
            plt.tight_layout()
            if save_fig:
                fig.savefig(f"{save_path}_{temp_id}_plane{pl}_stacked_peak{peak}.png", dpi=300)
            plt.close(fig)

if __name__ == '__main__':
    plt.style.use('latex')
    parser = argparse.ArgumentParser(description='Plot intensity vs. l for one or multiple voltages/planes.')
    parser.add_argument('csvfile', help='Path to tab-separated CSV file')
    parser.add_argument('--temp_id', type=str, required=True,
                        help='Temperature identifier string (e.g. "80K")')
    parser.add_argument('--voltage', nargs='+', type=float, required=True,
                        help='Voltage(s) in V')
    parser.add_argument('--plane', nargs='+', type=str, required=True,
                        help='Plane(s), e.g. "(h,3,l)"')
    parser.add_argument('--peak', type=int, choices=[1,2], default=1,
                        help='Which peak to plot (1 or 2)')
    parser.add_argument('--mode', choices=['overlay','side_by_side','stacked'], default='overlay',
                        help='Plotting mode')
    parser.add_argument('--save_fig', action='store_true', help='Save figures')
    parser.add_argument('--save_path', default='plot', help='Base path for saved figures')
    parser.add_argument('--rotate', action='store_true', help='Rotate plot -90°')
    args = parser.parse_args()

    df = load_data(args.csvfile)
    if len(args.voltage)==1 and len(args.plane)==1:
        plot_curve(df, args.temp_id, args.voltage[0], args.plane[0],
                   args.peak, rotate=args.rotate,
                   save_fig=args.save_fig, save_path=args.save_path)
    else:
        plot_dict = {}
        for v in args.voltage:
            rows = df[(df['Temperature (K)'] == args.temp_id) & (df['Voltage (V)'] == v)]
            for pl in args.plane:
                sel = rows[rows['Reciprocal Space Plane'] == pl.strip()]
                if sel.empty:
                    print(f"No data for {args.temp_id}, V={v}, plane={pl}")
                    continue
                row = sel.iloc[0]
                l_vals = np.array(row['l (np.array)'], dtype=float)
                intens = np.array(row['Intensity (np.array)'], dtype=float)
                plot_dict.setdefault(v, {})[pl] = (l_vals, intens)
        plots_function(plot_dict, args.plane, args.temp_id, args.peak,
                       args.voltage, args.mode, args.save_fig,
                       args.save_path, args.rotate)

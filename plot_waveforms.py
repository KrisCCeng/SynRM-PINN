import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse

# ================= Academic Plotting Standards =================
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'SimHei'], 
    'mathtext.fontset': 'stix', 
    'axes.labelsize': 14,
    'axes.titlesize': 14,
    'font.size': 14,
    'figure.dpi': 600,        
    'axes.unicode_minus': False,
    # 'svg.fonttype': 'none'  # Core: Ensure text is editable in Visio and improve file compatibility
})

# ================= Global Configuration Area =================
COLOR_1 = "#C22629"  # Voltage, current, torque color
COLOR_2 = "#1EA04D"  # Angle, flux linkage color

# [New Request]: Parameter label font size
ANNOT_FONT_SIZE = 16 

# [Visio Compatibility Optimization]: Darken grid color and width to prevent disappearance after import
GRID_COLOR = '#B0B0B0' 
GRID_LW = 0.8
# ===============================================

def calc_div(signal, force_val=None):
    """
    Automatically calculate oscilloscope Div scale (smart rounding logic)
    Requirement: Must be an integer if >= 1; retain one decimal place if < 1.
    """
    if force_val is not None:
        if force_val >= 1.0:
            return int(round(force_val))
        return round(force_val, 1)
    
    m = np.max(np.abs(signal))
    if m == 0:
        return 0.1
    
    raw_div = m / 2.5
    
    if raw_div >= 1.0:
        div = int(round(raw_div))
        if div == 0: div = 1
        return float(div)
    else:
        div = round(raw_div, 1)
        if div < 0.1:
            div = 0.1
        return div

def format_div(val):
    """Format Div text, force integer if >= 1, else one decimal place"""
    if val >= 1.0 and float(val).is_integer():
        return f"{int(val)}"
    else:
        return f"{val:.1f}"

def style_scope_ax(ax):
    """Apply oscilloscope style to underlying axes"""
    ax.set_ylim(-3.5, 3.5)
    ax.set_yticks([-3, -2, -1, 0, 1, 2, 3])
    ax.set_yticklabels(['', '', '', '0', '', '', ''])

    # Optimize grid settings to improve Visio compatibility
    ax.grid(True, axis='both', linestyle='-', color=GRID_COLOR, lw=GRID_LW, zorder=0)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)
    ax.spines['left'].set_color('#A0A0A0')
    ax.spines['bottom'].set_color('#A0A0A0')

    ax.plot(0, 0, marker='>', color='black', markersize=6, 
            transform=ax.get_yaxis_transform(), clip_on=False, zorder=5)
    
    ax.axhline(0, color='#808080', lw=1.2, zorder=1)

def add_annotation(ax, t, y_plot, text, color, find_max=True):
    """Smartly add label annotation with arrow"""
    if find_max:
        idx = np.argmax(y_plot)
    else:
        idx = np.argmin(y_plot)

    x_pt = t[idx]
    y_pt = y_plot[idx]

    offset_y = 15 if y_pt >= 0 else -20
    offset_x = 10
    
    if idx > len(t) * 0.8:
        offset_x = -80

    # Use global ANNOT_FONT_SIZE
    ax.annotate(text, xy=(x_pt, y_pt), xycoords='data',
                xytext=(offset_x, offset_y), textcoords='offset points',
                color=color, fontsize=ANNOT_FONT_SIZE, fontweight='bold',
                arrowprops=dict(arrowstyle="->", color=color, lw=1.2))

# ================= Core Plotting Functions =================

def plot_self_saturation(csv_path, axis_type, save_dir, is_sim):
    """Self-saturation experiment plotting"""
    df = pd.read_csv(csv_path)
    t = df.iloc[:, 0].values

    if axis_type == 'd_axis':
        v = df.iloc[:, 1].values
        psi = df.iloc[:, 3].values
        i = df.iloc[:, 5].values
        v_name, i_name, psi_name = r'$V_{d,ref}$', r'$i_d$', r'$\psi_d$'
    else:
        v = df.iloc[:, 2].values
        psi = df.iloc[:, 4].values
        i = df.iloc[:, 6].values
        v_name, i_name, psi_name = r'$V_{q,ref}$', r'$i_q$', r'$\psi_q$'

    div_v = calc_div(v)
    div_i = calc_div(i)
    div_psi = calc_div(psi)

    fig, axs = plt.subplots(2, 1, figsize=(7, 5), sharex=True)

    style_scope_ax(axs[0])
    axs[0].plot(t, v / div_v, color=COLOR_1, lw=1.8, zorder=3)
    add_annotation(axs[0], t, v / div_v, f"{v_name} [{format_div(div_v)} V/div.]", COLOR_1, find_max=True)
    
    if is_sim:
        theta_r = df.iloc[:, 8].values / 2.0  
        div_theta = calc_div(theta_r)
        axs[0].plot(t, theta_r / div_theta, color=COLOR_2, lw=1.8, zorder=2)
        add_annotation(axs[0], t, theta_r / div_theta, f"$\\theta_r$ [{format_div(div_theta)} rad/div.]", COLOR_2, find_max=False)

    style_scope_ax(axs[1])
    axs[1].plot(t, i / div_i, color=COLOR_1, lw=1.8, zorder=3)
    axs[1].plot(t, psi / div_psi, color=COLOR_2, lw=1.8, zorder=2)
    add_annotation(axs[1], t, i / div_i, f"{i_name} [{format_div(div_i)} A/div.]", COLOR_1, find_max=True)
    add_annotation(axs[1], t, psi / div_psi, f"{psi_name} [{format_div(div_psi)} Vs/div.]", COLOR_2, find_max=False)

    axs[1].set_xlabel('$t$ [s]')
    axs[1].set_xlim(t[0], t[-1]) 
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'waveform_{axis_type}.svg'), format='svg')
    plt.close(fig)

def plot_cross_saturation(csv_path, save_dir, is_sim):
    """Cross-saturation experiment plotting"""
    df = pd.read_csv(csv_path)
    t = df.iloc[:, 0].values
    ud, uq = df.iloc[:, 1].values, df.iloc[:, 2].values
    psid, psiq = df.iloc[:, 3].values, df.iloc[:, 4].values
    id_curr, iq_curr = df.iloc[:, 5].values, df.iloc[:, 6].values

    div_udq = calc_div(np.concatenate([ud, uq])) 
    div_id = calc_div(id_curr)
    div_psid = calc_div(psid)
    div_iq = calc_div(iq_curr)
    div_psiq = calc_div(psiq)

    n_subplots = 4 if is_sim else 3
    fig_height = 9.5 if is_sim else 7.125
    fig, axs = plt.subplots(n_subplots, 1, figsize=(7, fig_height), sharex=True)

    style_scope_ax(axs[0])
    axs[0].plot(t, ud / div_udq, color=COLOR_1, lw=1.8, zorder=3)
    axs[0].plot(t, uq / div_udq, color=COLOR_2, lw=1.8, zorder=2)
    add_annotation(axs[0], t, ud / div_udq, f"$V_{{d,ref}}$ [{format_div(div_udq)} V/div.]", COLOR_1, find_max=True)
    add_annotation(axs[0], t, uq / div_udq, f"$V_{{q,ref}}$ [{format_div(div_udq)} V/div.]", COLOR_2, find_max=False)

    style_scope_ax(axs[1])
    axs[1].plot(t, id_curr / div_id, color=COLOR_1, lw=1.8, zorder=3)
    axs[1].plot(t, psid / div_psid, color=COLOR_2, lw=1.8, zorder=2)
    add_annotation(axs[1], t, id_curr / div_id, f"$i_d$ [{format_div(div_id)} A/div.]", COLOR_1, find_max=True)
    add_annotation(axs[1], t, psid / div_psid, f"$\psi_d$ [{format_div(div_psid)} Vs/div.]", COLOR_2, find_max=False)

    style_scope_ax(axs[2])
    axs[2].plot(t, iq_curr / div_iq, color=COLOR_1, lw=1.8, zorder=3)
    axs[2].plot(t, psiq / div_psiq, color=COLOR_2, lw=1.8, zorder=2)
    add_annotation(axs[2], t, iq_curr / div_iq, f"$i_q$ [{format_div(div_iq)} A/div.]", COLOR_1, find_max=True)
    add_annotation(axs[2], t, psiq / div_psiq, f"$\psi_q$ [{format_div(div_psiq)} Vs/div.]", COLOR_2, find_max=False)

    if is_sim:
        Te = df.iloc[:, 7].values
        theta_r = df.iloc[:, 8].values / 2.0
        div_Te = calc_div(Te)
        div_theta = calc_div(theta_r, force_val=0.1)
        
        style_scope_ax(axs[3])
        axs[3].plot(t, Te / div_Te, color=COLOR_1, lw=1.8, zorder=3)
        axs[3].plot(t, theta_r / div_theta, color=COLOR_2, lw=1.8, zorder=2)
        add_annotation(axs[3], t, Te / div_Te, f"$T_e$ [{format_div(div_Te)} Nm/div.]", COLOR_1, find_max=True)
        add_annotation(axs[3], t, theta_r / div_theta, f"$\\theta_r$ [{format_div(div_theta)} rad/div.]", COLOR_2, find_max=False)
        
        axs[3].set_xlabel('$t$ [s]')
        axs[3].set_xlim(t[0], t[-1]) 
    else:
        axs[2].set_xlabel('$t$ [s]')
        axs[2].set_xlim(t[0], t[-1]) 

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'waveform_cross.svg'), format='svg')
    plt.close(fig)

def main(data_dir):
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"[Error] Dataset path not found: {data_dir}")

    d_axis_csv = glob.glob(os.path.join(data_dir, 'd_axis', '*.csv'))
    q_axis_csv = glob.glob(os.path.join(data_dir, 'q_axis', '*.csv'))
    cross_csv = glob.glob(os.path.join(data_dir, 'cross', '*.csv'))

    sample_files = d_axis_csv + q_axis_csv + cross_csv
    if not sample_files:
        print("[Error] No CSV files found in the specified path!")
        return
        
    df_sample = pd.read_csv(sample_files[0])
    is_sim = df_sample.shape[1] >= 9

    save_dir = os.path.join(data_dir, 'waveforms')
    os.makedirs(save_dir, exist_ok=True)

    if d_axis_csv:
        plot_self_saturation(d_axis_csv[0], 'd_axis', save_dir, is_sim)
    if q_axis_csv:
        plot_self_saturation(q_axis_csv[0], 'q_axis', save_dir, is_sim)
    if cross_csv:
        plot_cross_saturation(cross_csv[0], save_dir, is_sim)

    print(f"\n✅ All waveforms plotted successfully! Saved to: {save_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot oscilloscope-style waveforms for training datasets")
    parser.add_argument("--data_dir", type=str, required=True, help="Dataset main directory path")
    args = parser.parse_args()
    main(args.data_dir)

# python plot_waveforms.py --data_dir data/simulation/4000ideal
# python plot_waveforms.py --data_dir data/processed/Exdataset3
import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
import pandas as pd

# ================= Global Academic Plotting Standards =================
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'SimHei'], 
    'mathtext.fontset': 'stix', 
    'axes.labelsize': 20,
    'axes.titlesize': 20,
    'font.size': 20,
    'legend.fontsize': 20,
    'xtick.labelsize': 20,
    'ytick.labelsize': 20,
    'axes.linewidth': 1.0,
    'figure.dpi': 300,        
    'axes.unicode_minus': False
})

def plot_flux_surfaces(id_grid, iq_grid, psid_pred, psiq_pred, train_data_dict=None, mode='surface', save_dir=None):
    fig = plt.figure(figsize=(13, 6))
    
    color_map = {
        'd_axis': '#D62728', 
        'q_axis': '#1F77B4', 
        'cross':  '#A0A0A0'  
    }
    plot_order = ['cross', 'q_axis', 'd_axis']

    # ================= Plot D-axis =================
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.set_title('$\psi_d$ (D-axis Flux)')
    ax1.set_xlabel('$i_d$ (A)')
    ax1.set_ylabel('$i_q$ (A)')
    ax1.set_zlabel('$\psi_d$ (Vs)')
    ax1.view_init(elev=25, azim=-135) 

    if mode == 'surface':
        # Core modification: Add rasterized=True to rasterize complex surfaces, preventing Visio/Word from crashing.
        surf1 = ax1.plot_surface(id_grid, iq_grid, psid_pred, cmap='RdBu_r', alpha=0.85, 
                                 edgecolor='none', antialiased=True, rasterized=True)
        fig.colorbar(surf1, ax=ax1, shrink=0.5, aspect=12, pad=0.1)
        
    elif mode == 'data' and train_data_dict:
        for label in plot_order:
            if label in train_data_dict:
                id_d, iq_d, pd_d, pq_d = train_data_dict[label]
                c = color_map.get(label, 'black')
                
                z_offset = 0.08 if label in ['d_axis', 'q_axis'] else 0.0
                alpha_val = 1.0 if label in ['d_axis', 'q_axis'] else 0.3
                z_ord = 10 if label in ['d_axis', 'q_axis'] else 1
                
                # Scatter plots can lag if there are too many points, add rasterized=True.
                ax1.scatter(id_d, iq_d, pd_d + z_offset, c=c, marker='o', s=15, 
                            alpha=alpha_val, depthshade=False, zorder=z_ord, label=label, rasterized=True)
        ax1.legend(loc='upper left', bbox_to_anchor=(0.0, 1.0))

    # ================= Plot Q-axis =================
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.set_title('$\psi_q$ (Q-axis Flux)')
    ax2.set_xlabel('$i_d$ (A)')
    ax2.set_ylabel('$i_q$ (A)')
    ax2.set_zlabel('$\psi_q$ (Vs)')
    ax2.view_init(elev=25, azim=225) 

    if mode == 'surface':
        # Core modification: Add rasterized=True
        surf2 = ax2.plot_surface(id_grid, iq_grid, psiq_pred, cmap='RdBu_r', alpha=0.85, 
                                 edgecolor='none', antialiased=True, rasterized=True)
        fig.colorbar(surf2, ax=ax2, shrink=0.5, aspect=12, pad=0.1)
        
    elif mode == 'data' and train_data_dict:
        for label in plot_order:
            if label in train_data_dict:
                id_d, iq_d, pd_d, pq_d = train_data_dict[label]
                c = color_map.get(label, 'black')
                
                z_offset = 0.08 if label in ['d_axis', 'q_axis'] else 0.0
                alpha_val = 1.0 if label in ['d_axis', 'q_axis'] else 0.3
                z_ord = 10 if label in ['d_axis', 'q_axis'] else 1
                
                # Add rasterized=True
                ax2.scatter(id_d, iq_d, pq_d + z_offset, c=c, marker='o', s=15, 
                            alpha=alpha_val, depthshade=False, zorder=z_ord, label=label, rasterized=True)
        ax2.legend(loc='upper left', bbox_to_anchor=(0.0, 1.0))

    plt.tight_layout()
    
    if save_dir:
        filename = f"flux_3d_{mode}.svg"
        # Core modification: Set a very high DPI (600). This does not affect vector text (remains lossless) but makes rasterized surfaces extremely clear.
        plt.savefig(os.path.join(save_dir, filename), bbox_inches='tight', format='svg', dpi=600)
        plt.close(fig)
    else:
        plt.show()


def plot_error_heatmaps(id_grid, iq_grid, err_d, err_q, save_dir=None, Derror_limit=0.04, Qerror_limit=0.04):
    """
    Plot error heatmaps (Fully vector academic standard version, unified Colorbar range)
    """
    fig, axs = plt.subplots(1, 2, figsize=(13, 5))
    
    # Cancel previous dynamic extreme values, uniformly use the passed error_limit.
    Dvmin = -Derror_limit
    Dvmax = Derror_limit
    Qvmin = -Qerror_limit
    Qvmax = Qerror_limit
    
    # Plot D-axis error
    im1 = axs[0].pcolormesh(id_grid, iq_grid, err_d, cmap='RdBu_r', vmin=Dvmin, vmax=Dvmax, rasterized=True)
    axs[0].set_title('$\psi_d$ Error (Vs)')
    axs[0].set_xlabel('$i_d$ (A)')
    axs[0].set_ylabel('$i_q$ (A)')
    # Add extend='both': If the actual error exceeds the limit, arrows will indicate it at both ends of the Colorbar.
    fig.colorbar(im1, ax=axs[0], format="%.3f", extend='both')
    
    # Plot Q-axis error
    im2 = axs[1].pcolormesh(id_grid, iq_grid, err_q, cmap='RdBu_r', vmin=Qvmin, vmax=Qvmax, rasterized=True)
    axs[1].set_title('$\psi_q$ Error (Vs)')
    axs[1].set_xlabel('$i_d$ (A)')
    axs[1].set_ylabel('$i_q$ (A)')
    fig.colorbar(im2, ax=axs[1], format="%.3f", extend='both')
    
    plt.tight_layout()
    if save_dir:
        plt.savefig(os.path.join(save_dir, 'error_heatmap.svg'), bbox_inches='tight', format='svg', dpi=600)
        plt.close(fig)
    else:
        plt.show()


def plot_training_logs(log_dict, save_dir=None):
    """
    Plot and compare total training loss and MSE loss of multiple models (Advanced academic version: raw data + adaptive local zoom)
    """

    # Strictly follow academic color palettes and linestyles.
    color_palette = {'ANN': '#1F77B4', 'SCN': '#FF7F0E', 'PINN': '#D62728'}
    linestyle_palette = {'ANN': ':', 'SCN': '--', 'PINN': '-'}

    fig, axs = plt.subplots(1, 2, figsize=(14, 5.5))

    # Create inset axes for local zoom.
    axins0 = axs[0].inset_axes([0.45, 0.45, 0.5, 0.4]) 
    axins1 = axs[1].inset_axes([0.45, 0.45, 0.5, 0.4])
    
    # Store data in the zoom range (150-200) to calculate adaptive y-axis limits.
    zoom_data_total = []
    zoom_data_mse = []

    for model_name, log_path in log_dict.items():
        try:
            df = pd.read_csv(log_path)
            if 'epoch' not in df.columns or 'train_loss' not in df.columns:
                continue
            
            label_name = 'PINN' if 'PINN' in model_name else ('SCN' if 'SCN' in model_name else 'ANN')
            c = color_palette.get(label_name, '#333333')
            ls = linestyle_palette.get(label_name, '-')
            epochs = df['epoch'].values

            # ---------------- 1. Plot Total Training Loss ----------------
            train_loss = df['train_loss'].values
            axs[0].plot(epochs, train_loss, color=c, label=label_name, linewidth=2.0, linestyle=ls)
            axins0.plot(epochs, train_loss, color=c, linewidth=2.0, linestyle=ls)
            
            # Extract data points between epochs 150-200.
            mask = (epochs >= 150) & (epochs <= 200)
            if any(mask):
                zoom_data_total.extend(train_loss[mask])

            # ---------------- 2. Plot MSE Loss (Data Fit) ----------------
            if 'mse_loss' in df.columns:
                mse_loss = df['mse_loss'].values
                axs[1].plot(epochs, mse_loss, color=c, label=label_name, linewidth=2.0, linestyle=ls)
                axins1.plot(epochs, mse_loss, color=c, linewidth=2.0, linestyle=ls)
                
                if any(mask):
                    zoom_data_mse.extend(mse_loss[mask])

        except Exception as e:
            print(f"  [Warning] Cannot read or plot log for {model_name}: {e}")

    # ================= Adaptively adjust y-axis limits of inset axes =================
    def set_adaptive_ylim(ax_ins, data_list, margin_ratio=0.15):
        if not data_list: return
        ymin, ymax = min(data_list), max(data_list)
        delta = ymax - ymin
        # If data overlaps completely (delta=0), provide a default range to prevent ticks from disappearing.
        if delta == 0: delta = ymin * 0.1
        ax_ins.set_ylim(ymin - delta * margin_ratio, ymax + delta * margin_ratio)

    set_adaptive_ylim(axins0, zoom_data_total)
    set_adaptive_ylim(axins1, zoom_data_mse)

    # ================= Style Beautification and Axis Configuration =================
    titles = ['Total Training Loss Comparison', 'MSE Loss Comparison (Data Fit)']
    for i, ax in enumerate(axs):
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_yscale('log')
        ax.set_xlabel('Epoch', fontweight='bold')
        ax.set_title(titles[i], fontweight='bold', pad=15)
        ax.grid(True, which="major", ls="--", alpha=0.5)
        # [Modification 1]: Removed ax.legend(...) here to avoid generating legends inside subplots.

    axs[0].set_ylabel('Total Loss (Log Scale)', fontweight='bold')
    axs[1].set_ylabel('MSE Loss (Log Scale)', fontweight='bold')

    # Configure general style for inset axes.
    for axins in [axins0, axins1]:
        axins.set_xlim(150, 200)
        axins.set_yscale('linear') 
        axins.grid(True, ls=":", alpha=0.5)
        axins.tick_params(axis='both', which='major', labelsize=12)
        axins.yaxis.get_major_formatter().set_powerlimits((0,1)) 

    # Draw connecting lines.
    mark_inset(axs[0], axins0, loc1=3, loc2=4, fc="none", ec="gray", alpha=0.4, linestyle='--')
    mark_inset(axs[1], axins1, loc1=3, loc2=4, fc="none", ec="gray", alpha=0.4, linestyle='--')

    # ================= [Modification 2]: Add global unified legend =================
    # Extract line handles and labels from the first plot.
    handles, labels = axs[0].get_legend_handles_labels()
    
    # Place a horizontally aligned legend (ncol=3) at the top of the entire Figure.
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0.92), 
               ncol=3, frameon=True, edgecolor='black', fontsize=16)

    # Adjust subplot layout, leaving 8% blank space at the top specifically for the legend.
    plt.tight_layout(rect=[0, 0, 1, 0.92])

    if save_dir:
        # [Modification 3]: Add bbox_inches='tight' to prevent legend edges from being cropped during saving.
        plt.savefig(os.path.join(save_dir, 'training_logs_comparison_adaptive.svg'), 
                    format='svg', dpi=600, bbox_inches='tight')
        plt.close(fig)
        print("✅ Training logs plot saved (Legend occlusion fixed, global legend enabled).")
    else:
        plt.show()

def plot_pinn_loss_components(log_path, model_name, save_dir=None):
    """
    Plot weight evolution and 'effective loss contribution' of the PINN model (Supports SymLog for negative values)
    """
    import matplotlib.pyplot as plt
    import pandas as pd
    import os
    
    df = pd.read_csv(log_path)
    weight_cols = [c for c in df.columns if c.startswith('w_')]

    if not weight_cols:
        return 

    fig, axs = plt.subplots(1, 2, figsize=(13, 5))

    # ---- Subplot 1: Pure Weights Evolution ----
    for w_col in weight_cols:
        axs[0].plot(df['epoch'], df[w_col], label=w_col, linewidth=1.5, alpha=0.85)
        
    axs[0].set_title(f'[{model_name}] Loss Weights Evolution')
    axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel('Weight Value (SymLog Scale)')
    axs[0].set_yscale('symlog', linthresh=1e-3)
    axs[0].legend(fontsize=9)
    axs[0].grid(True, linestyle='--', alpha=0.6)

    # ---- Subplot 2: Effective Loss Contributions (Weight * Loss) ----
    has_effective = False
    for w_col in weight_cols:
        loss_col = w_col.replace('w_', 'loss_') 
        
        if loss_col in df.columns:
            # You can also plot (df[w_col] * df[loss_col] + log_sigma term) here to see the true negative contribution including the regularization term.
            # But academia usually shows df[w_col] * df[loss_col] to demonstrate the pure pull of physical constraints.
            effective_loss = df[w_col] * df[loss_col]
            axs[1].plot(df['epoch'], effective_loss, label=f"{w_col} $\\times$ {loss_col}", linewidth=1.5, alpha=0.85)
            has_effective = True

    if has_effective:
        axs[1].set_title('Effective Loss Contributions (Weight $\\times$ Loss)')
        axs[1].set_xlabel('Epoch')
        axs[1].set_ylabel('Effective Loss Magnitude (SymLog Scale)')
        axs[1].set_yscale('symlog', linthresh=1e-6) # Enable symlog to accommodate extreme values as well.
        axs[1].legend(fontsize=9)
        axs[1].grid(True, linestyle='--', alpha=0.6)
    else:
        axs[1].text(0.5, 0.5, 'Raw loss columns not found in log.',
                    ha='center', va='center', color='gray', transform=axs[1].transAxes)
        axs[1].axis('off')

    plt.tight_layout()
    if save_dir:
        plt.savefig(os.path.join(save_dir, 'pinn_components_evolution.svg'), format='svg')
        plt.close(fig)
    else:
        plt.show()
import os
import glob
import re
import yaml
import torch
import numpy as np
import pandas as pd
import argparse
from tabulate import tabulate
from src import SymmetricSynRMModel, PolynomialModel, LookUpTableModel, MotorDataset
from src.utils import get_device, set_seed
from src.visualization import plot_flux_surfaces, plot_error_heatmaps

I_MAX = 4.2144
GRID_STEPS = 40
# Simplified plotting modes: surface (3D surface), data (individual training data scatter plots)
DRAW_MODES = ['surface', 'data'] 

def load_experimental_data_dict(data_dir):
    """Dynamically load scatter data based on the given dataset directory"""
    data_dict = {}
    folders = ["d_axis", "q_axis", "cross"]
    for folder_name in folders:
        folder_path = os.path.join(data_dir, folder_name)
        if os.path.exists(folder_path):
            dataset = MotorDataset([folder_path])
            if len(dataset) > 0:
                data_dict[folder_name] = (
                    dataset.idiq_now[:, 0], dataset.idiq_now[:, 1],
                    dataset.targets[:, 0], dataset.targets[:, 1]
                )
    return data_dict

def get_ground_truth(id_grid, iq_grid, is_simulation):
    """Get the ground truth flux linkage based on data type, introducing the fused reference cache mechanism"""
    if is_simulation:
        print("[Info] Simulation data detected. Using PolynomialModel as Ground Truth.")
        cache_file = "data/simulation/simulation_gt_cache.npz"
        
        if os.path.exists(cache_file):
            print(f"  -> Loading cached Polynomial Ground Truth from {cache_file}")
            data = np.load(cache_file)
            return data['Psid_gt'], data['Psiq_gt']
            
        print("  -> Calculating Numerical Inverse... This might take a while.")
        poly_model = PolynomialModel()
        psid_flat, psiq_flat = poly_model.numerical_inverse(id_grid.ravel(), iq_grid.ravel())
        psid_gt = psid_flat.reshape(id_grid.shape)
        psiq_gt = psiq_flat.reshape(iq_grid.shape)
        
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        np.savez(cache_file, Psid_gt=psid_gt, Psiq_gt=psiq_gt)
        print(f"  -> Cache saved to {cache_file}")
        return psid_gt, psiq_gt
    else:
        # Experimental data evaluation prioritizes reading the fused and smoothed reference flux linkage
        cache_file = os.path.join("data", "processed", "filtered_gt_cache.npz")
        
        if os.path.exists(cache_file):
            print(f"[Info] Experimental data detected. Using Filtered Reference Ground Truth.")
            print(f"  -> Loading reference from {cache_file}")
            data = np.load(cache_file)
            
            # Shape safety check
            if data['Psid_gt'].shape != id_grid.shape:
                raise ValueError(f"Cached grid size {data['Psid_gt'].shape} does not match current GRID_STEPS setting {id_grid.shape}! Please re-run generate_filtered_gt.py")
                
            return data['Psid_gt'], data['Psiq_gt']
        else:
            print("[Warning] Reference flux linkage file (filtered_gt_cache.npz) not found!")
            print("  -> Falling back to raw LookUpTable as the ground truth.")
            lut_model = LookUpTableModel()
            return lut_model.predict(id_grid, iq_grid)

def main(args):
    # --- 1. Load YAML configuration file ---
    config = {}
    if args.config:
        config_path = args.config
        if not os.path.exists(config_path) and not config_path.startswith('configs'):
            config_path = os.path.join('configs', args.config)
            
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"[Error] Configuration file not found: '{args.config}' or '{config_path}'!")
            
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f) or {}
        print(f"✅ Successfully loaded configuration file: {config_path}")
        
    # --- 2. Extract configuration parameters ---
    data_cfg = config.get('data', {})
    data_dir = data_cfg.get('data_dir', args.data_dir)
    i_norm = data_cfg.get('i_norm', 4.0)
    psi_norm = data_cfg.get('psi_norm', 1.5)
    default_hidden_dim = config.get('training', {}).get('hidden_dim', [6, 4])

    if data_dir is None:
        raise ValueError("Must specify data_dir via command line --data_dir or in the yaml file under data!")

    set_seed(42)
    device = get_device()
    dataset_name = os.path.basename(os.path.normpath(data_dir))
    is_simulation = 'simulation' in data_dir.lower()
    
    models_dir = os.path.join("outputs", dataset_name, "models")
    results_dir = os.path.join("outputs", dataset_name, "evaluation_results")
    os.makedirs(results_dir, exist_ok=True)
    
    # --- 3. Prepare grid and data ---
    id_vals = np.linspace(-I_MAX, I_MAX, GRID_STEPS)
    iq_vals = np.linspace(-I_MAX, I_MAX, GRID_STEPS)
    ID_grid, IQ_grid = np.meshgrid(id_vals, iq_vals)
    inp_tensor = torch.tensor(np.stack([ID_grid.ravel(), IQ_grid.ravel()], axis=1), dtype=torch.float32).to(device)

    exp_data_dict = load_experimental_data_dict(data_dir)
    Psid_gt, Psiq_gt = get_ground_truth(ID_grid, IQ_grid, is_simulation)

    # From log file plotting module (keep as needed)
    from src.visualization import plot_training_logs, plot_pinn_loss_components
    logs_dir = os.path.join("outputs", dataset_name, "logs")
    log_files = glob.glob(os.path.join(logs_dir, "*.csv"))
    log_dict = {os.path.splitext(os.path.basename(p))[0]: p for p in log_files}

    if log_dict:
        print("\n[Drawing] Generating global training logs comparison plot...")
        plot_training_logs(log_dict, save_dir=results_dir)

    # --- 4. Scan and evaluate all models ---
    model_files = glob.glob(os.path.join(models_dir, "*.pth"))
    if not model_files:
        print(f"[Error] No model files found in {models_dir}")
        return

    metrics_list = []
    
    for model_path in model_files:
        model_name = os.path.splitext(os.path.basename(model_path))[0]
        print(f"\n[Evaluating] {model_name} ...")
        
        # [Modification]: Parse model mode - Distinguish between ANN, SCN, PINN
        if "PINN" in model_name:
            mode = "PINN"
        elif "SCN" in model_name:
            mode = "SCN"
        else:
            mode = "ANN"
        
        match = re.search(r'Net([\d_]+)', model_name)
        if match:
            hd_str = match.group(1).strip('_')
            if '_' in hd_str:
                current_hidden_dim = [int(x) for x in hd_str.split('_')]
            else:
                current_hidden_dim = [int(hd_str), int(hd_str)]
        else:
            current_hidden_dim = default_hidden_dim
        
        # [Modification]: Instantiate corresponding model class based on mode
        # (Don't forget to import TraditionalANN from src at the top)
        from src import SymmetricSynRMModel, TraditionalANN, PolynomialModel, LookUpTableModel, MotorDataset
        
        if mode == "ANN":
            model = TraditionalANN(
                i_norm=i_norm, 
                psi_norm=psi_norm,
                hidden_dim=current_hidden_dim
            ).to(device)
        else:
            model = SymmetricSynRMModel(
                mode=mode, 
                i_norm=i_norm, 
                psi_norm=psi_norm,
                hidden_dim=current_hidden_dim
            ).to(device)
        
        # Compatibility loading
        checkpoint = torch.load(model_path, map_location=device)
        
        train_time_val = "N/A"
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
            train_time_val = round(checkpoint.get('train_time', 0.0), 2)
        else:
            model.load_state_dict(checkpoint)
            
        model.eval()
        
        with torch.no_grad():
            pred = model.forward_flux(inp_tensor).cpu().numpy()
            Psid_pred = pred[:, 0].reshape(ID_grid.shape)
            Psiq_pred = pred[:, 1].reshape(ID_grid.shape)
            
        # =================== Core Metric Calculation ===================
        err_d = Psid_pred - Psid_gt
        err_q = Psiq_pred - Psiq_gt
        
        # Calculate MAPE (Mean Absolute Percentage Error)
        # Set threshold epsilon to filter out unstable points where denominator is close to 0, set to 5e-3 (0.005 Vs) here
        epsilon = 5e-3
        
        valid_idx_d = np.abs(Psid_gt) > epsilon
        mape_d = np.mean(np.abs(err_d[valid_idx_d] / Psid_gt[valid_idx_d])) * 100 if np.any(valid_idx_d) else 0.0
        
        valid_idx_q = np.abs(Psiq_gt) > epsilon
        mape_q = np.mean(np.abs(err_q[valid_idx_q] / Psiq_gt[valid_idx_q])) * 100 if np.any(valid_idx_q) else 0.0

        metrics_list.append({
            'Model': model_name,
            'Train Time (s)': train_time_val,
            'Psid MAE': np.mean(np.abs(err_d)), 
            'Psid RMSE': np.sqrt(np.mean(err_d**2)),
            'Psid MAPE (%)': mape_d,             # Added D-axis MAPE
            'Psiq MAE': np.mean(np.abs(err_q)), 
            'Psiq RMSE': np.sqrt(np.mean(err_q**2)),
            'Psiq MAPE (%)': mape_q              # Added Q-axis MAPE
        })
        # ====================================================
        
        # --- 5. Plotting and output ---
        save_sub = os.path.join(results_dir, model_name)
        os.makedirs(save_sub, exist_ok=True)
        
        plot_error_heatmaps(ID_grid, IQ_grid, err_d, err_q, save_dir=save_sub)
        
        for plot_mode in DRAW_MODES:
            plot_flux_surfaces(
                ID_grid, IQ_grid, Psid_pred, Psiq_pred, 
                train_data_dict=exp_data_dict,
                mode=plot_mode,
                save_dir=save_sub
            )
            
        if mode == "PINN" and model_name in log_dict:
            plot_pinn_loss_components(log_dict[model_name], model_name, save_dir=save_sub)

    # ====================================================
    # 7. Evaluate traditional polynomial fitting model from literature (Traditional Polynomial Model)
    # ====================================================
    if not is_simulation:
        model_name = "Traditional_Polynomial"
        print(f"\n[Evaluating] {model_name} ...")
        
        poly_eval_model = PolynomialModel()
        
        print("  -> Fitting polynomial coefficients using experimental data (LLS)...")
        poly_eval_model.fit(exp_data_dict)
            
        print("  -> Calculating Numerical Inverse for polynomial model...")
        # Calculate numerical inverse
        Psid_poly_flat, Psiq_poly_flat = poly_eval_model.numerical_inverse(ID_grid.ravel(), IQ_grid.ravel())
        Psid_poly_pred = Psid_poly_flat.reshape(ID_grid.shape)
        Psiq_poly_pred = Psiq_poly_flat.reshape(ID_grid.shape)
        
        err_d_poly = Psid_poly_pred - Psid_gt
        err_q_poly = Psiq_poly_pred - Psiq_gt
        
        epsilon = 5e-3
        valid_idx_d = np.abs(Psid_gt) > epsilon
        mape_d_poly = np.mean(np.abs(err_d_poly[valid_idx_d] / Psid_gt[valid_idx_d])) * 100 if np.any(valid_idx_d) else 0.0
        
        valid_idx_q = np.abs(Psiq_gt) > epsilon
        mape_q_poly = np.mean(np.abs(err_q_poly[valid_idx_q] / Psiq_gt[valid_idx_q])) * 100 if np.any(valid_idx_q) else 0.0

        metrics_list.append({
            'Model': model_name,
            'Train Time (s)': np.nan,  # Traditional formula fitting does not require neural network training, marked as NaN
            'Psid MAE': np.mean(np.abs(err_d_poly)), 
            'Psid RMSE': np.sqrt(np.mean(err_d_poly**2)),
            'Psid MAPE (%)': mape_d_poly,
            'Psiq MAE': np.mean(np.abs(err_q_poly)), 
            'Psiq RMSE': np.sqrt(np.mean(err_q_poly**2)),
            'Psiq MAPE (%)': mape_q_poly
        })
        
        save_sub_poly = os.path.join(results_dir, model_name)
        os.makedirs(save_sub_poly, exist_ok=True)
        
        plot_error_heatmaps(ID_grid, IQ_grid, err_d_poly, err_q_poly, save_dir=save_sub_poly)
        
        for plot_mode in DRAW_MODES:
            plot_flux_surfaces(
                ID_grid, IQ_grid, Psid_poly_pred, Psiq_poly_pred, 
                train_data_dict=exp_data_dict,
                mode=plot_mode,
                save_dir=save_sub_poly
            )
    else:
        print("\n[Info] Simulation data detected. Skipping Polynomial Model evaluation as it acts as the ground truth.")

    # --- 8. Generate overall report ---
    if metrics_list:
        df = pd.DataFrame(metrics_list)
        df.to_csv(os.path.join(results_dir, "evaluation_metrics.csv"), index=False)
        print("\n" + "="*80 + "\nEVALUATION REPORT\n" + "="*80)
        # floatfmt parameter aligns all float formats
        print(tabulate(df, headers='keys', tablefmt='psql', floatfmt=".4f"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None, help="YAML configuration file path")
    parser.add_argument("--data_dir", type=str, default=None, help="Corresponding dataset main directory path for evaluation")
    args = parser.parse_args()
    main(args)
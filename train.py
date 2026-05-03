# train.py (Key code modifications)
import os
import argparse
import yaml
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
from src import (
    MotorDataset, 
    create_dataloader, 
    SymmetricSynRMModel, 
    TraditionalANN,
    PCGrad, 
    StaticLossWeights,
    train_one_epoch
)
from src.utils import set_seed, ExperimentLogger, get_device



def main(args):
    # --- 1. Parameter and path setup ---
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
            
    mode = config.get('mode', args.mode)
    seed = config.get('seed', 42)
    
    data_cfg = config.get('data', {})
    data_dir = data_cfg.get('data_dir', args.data_dir)
    batch_size = data_cfg.get('batch_size', args.batch_size)
    i_norm = data_cfg.get('i_norm', 4.0)
    psi_norm = data_cfg.get('psi_norm', 1.5)
    
    train_cfg = config.get('training', {})
    epochs = train_cfg.get('epochs', args.epochs)
    lr = float(train_cfg.get('lr', args.lr))
    dt = float(train_cfg.get('dt', args.dt))
    hidden_dim = train_cfg.get('hidden_dim', [64, 64]) # Default list
    w_phys = float(train_cfg.get('w_phys', getattr(args, 'w_phys', 0.25)))
    
    # Uniformly convert to list for passing, and generate string for naming
    if isinstance(hidden_dim, int):
        hidden_dim = [hidden_dim, hidden_dim]
    hidden_str = "_".join(map(str, hidden_dim)) # For example "64_32"
    
    if data_dir is None:
        raise ValueError("Must specify data_dir via command line --data_dir or in the yaml file under data!")
        
    dataset_name = os.path.basename(os.path.normpath(data_dir))
    
    save_dir = os.path.join("outputs", dataset_name, "models")
    log_dir = os.path.join("outputs", dataset_name, "logs")
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    set_seed(seed)
    device = get_device()
    
    # --- 2. Data Loading ---
    folder_list = ["d_axis", "q_axis", "cross"]
    train_folders = [os.path.join(data_dir, f) for f in folder_list]
    
    print(f"[{dataset_name}] Loading data from: {train_folders}")
    train_loader = create_dataloader(train_folders, batch_size=batch_size, shuffle=True)

    # --- 3. Model Initialization ---
    if mode == "ANN":
        print("[Model] Initializing Traditional Black-box ANN...")
        model = TraditionalANN(
            i_norm=i_norm, 
            psi_norm=psi_norm,
            hidden_dim=hidden_dim
        ).to(device)
    else:
        print(f"[Model] Initializing Structure-Constrained Model (Mode: {mode})...")
        model = SymmetricSynRMModel(
            dt=dt, 
            mode=mode,
            i_norm=i_norm, 
            psi_norm=psi_norm,
            hidden_dim=hidden_dim
        ).to(device)
    
    # Loss weight control
    if mode == "PINN":
        # [Modification]: Use the read w_phys
        loss_weighter = StaticLossWeights(w_mse=1.0, w_phys=w_phys).to(device)
        params = model.parameters() 
        # Add weight identifier at the end of filename to avoid overwriting in ablation studies
        weight_str = f"_W{w_phys}"
    else:
        loss_weighter = None
        params = model.parameters()
        weight_str = "" # ANN and SCN do not have physical weights, no suffix needed

    # Create native Adam optimizer
    base_optimizer = optim.Adam(params, lr=lr)
    
    if mode == "PINN":
        optimizer = PCGrad(base_optimizer)
        scheduler_optim = base_optimizer
    else:
        optimizer = base_optimizer
        scheduler_optim = base_optimizer

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(scheduler_optim, mode='min', factor=0.5, patience=20)
    criterion = nn.MSELoss()
    
    # [Modification]: Add weight_str to log and model naming
    log_save_name = f"{mode}_Net{hidden_str}_Bz{batch_size}_Ep{epochs}{weight_str}.csv"
    logger = ExperimentLogger(save_dir=log_dir, filename=log_save_name)

    # --- 4. Training and Saving ---
    # Filename includes underscore-concatenated layer sizes, along with the weight suffix
    model_save_name = f"{mode}_Net{hidden_str}_Bz{batch_size}_Ep{epochs}{weight_str}.pth"
    model_save_path = os.path.join(save_dir, model_save_name)
    
    print(f"Start training... Model will be saved as {model_save_name}")
    best_loss = float('inf')

    start_time = time.time() # Record training start time
    
    for epoch in range(1, epochs + 1):
        avg_loss, avg_mse, weights = train_one_epoch(
            model, train_loader, optimizer, loss_weighter, criterion, device
        )
        scheduler.step(avg_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        log_data = {
            'train_loss': avg_loss,
            'mse_loss': avg_mse,
            'lr': current_lr
        }
        if weights:
            log_data.update(weights)
        
        logger.log_metrics(epoch, log_data)
        
        # Save the best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), model_save_path)
            
        if epoch % 10 == 0:
            log_str = f"Ep {epoch} | Loss: {avg_loss:.5f} | MSE: {avg_mse:.5f} | LR: {current_lr:.1e}"
            if mode == "PINN" and weights:
                # Print weights and corresponding raw PDE1 loss for intuitive monitoring
                log_str += f" | w_PDE1: {weights.get('w_pde1', 0):.2f} | raw_PDE1: {weights.get('loss_pde1', 0):.5f}"
            print(log_str)

    total_time = time.time() - start_time # Calculate total time
    logger.save()
    print(f"Training finished in {total_time:.2f} seconds. Best Loss: {best_loss:.6f}")

    if os.path.exists(model_save_path):
        best_state = torch.load(model_save_path)
        torch.save({
            'state_dict': best_state,
            'train_time': total_time
        }, model_save_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--data_dir", type=str, default=None, help="Data main directory path (e.g., data/simulation/sim1)")
    parser.add_argument("--mode", type=str, default="SCN", choices=["ANN", "SCN", "PINN"])
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--hidden_dim", type=int, default=64, help="Network hidden layer dimensions")
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--dt", type=float, default=1e-4)
    # [Added] This line of code
    parser.add_argument("--w_phys", type=float, default=0.25, help="Weight of the PINN physical loss")
    args = parser.parse_args()
    main(args)
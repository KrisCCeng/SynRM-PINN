import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader

class MotorDataset(Dataset):
    def __init__(self, folder_names, root_dir=None):
        """
        Args:
            folder_names (list): List of data folder names, e.g., ["processed_data"]
            root_dir (str): Project root directory, defaults to current working directory
        """
        self.samples = []
        
        # Temporary containers
        all_inputs = []
        all_targets = []
        all_idiq_now = []
        all_idiq_next = []
        all_we = []

        if root_dir is None:
            root_dir = os.getcwd()

        print(f"[Dataset] Start loading data, root directory: {root_dir}")
        
        for folder_name in folder_names:
            folder_path = os.path.join(root_dir, folder_name)
            if not os.path.exists(folder_path):
                print(f"[Dataset] Warning: Folder {folder_path} does not exist, skipping.")
                continue
                
            file_list = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
            
            for file_name in file_list:
                file_path = os.path.join(folder_path, file_name)
                try:
                    # Read CSV
                    df = pd.read_csv(file_path)
                    # Assume column structure based on original code: [Timestamp, ud, uq, psid, psiq, id, iq, speed (optional)]
                    # Take columns 1 to 8 (index 1:8)
                    val = df.values[:, 1:8]
                    
                    N = val.shape[0]
                    if N < 2: continue
                    
                    # --- Core logic: Slicing within file to ensure differential physical meaning ---
                    # Time k
                    curr_data = val[:-1, :]
                    # Time k+1
                    next_data = val[1:, :]
                    
                    # Extract physical quantities (column indices inferred from MotorPhysicsModel.py)
                    # columns in 'val': 0:ud, 1:uq, 2:psid, 3:psiq, 4:id, 5:iq
                    ud = curr_data[:, 0]
                    uq = curr_data[:, 1]
                    psid = curr_data[:, 2]
                    psiq = curr_data[:, 3]
                    
                    id_now = curr_data[:, 4]
                    iq_now = curr_data[:, 5]
                    
                    id_next = next_data[:, 4]
                    iq_next = next_data[:, 5]
                    
                    # Speed we (assuming column 6 in val, pad with 0 if missing)
                    if val.shape[1] > 6:
                        we_data = curr_data[:, 6]
                    else:
                        we_data = np.zeros_like(id_now)

                    # Store in lists
                    all_inputs.append(np.stack([ud, uq], axis=1))
                    all_targets.append(np.stack([psid, psiq], axis=1))
                    all_idiq_now.append(np.stack([id_now, iq_now], axis=1))
                    all_idiq_next.append(np.stack([id_next, iq_next], axis=1))
                    all_we.append(we_data.reshape(-1, 1))
                    
                except Exception as e:
                    print(f"[Dataset] Error reading file {file_name}: {e}")
                    continue

        # Merge all experimental data
        if len(all_inputs) > 0:
            self.inputs = np.concatenate(all_inputs, axis=0)
            self.targets = np.concatenate(all_targets, axis=0)
            self.idiq_now = np.concatenate(all_idiq_now, axis=0)
            self.idiq_next = np.concatenate(all_idiq_next, axis=0)
            self.we = np.concatenate(all_we, axis=0)
            print(f"[Dataset] Loading complete: {len(self.inputs)} sample pairs (t, t+1) loaded.")
        else:
            print("[Dataset] Error: No data loaded.")
            self.inputs = np.array([])

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        # Return Tensor float32
        return (torch.tensor(self.inputs[idx], dtype=torch.float32),
                torch.tensor(self.targets[idx], dtype=torch.float32),
                torch.tensor(self.idiq_now[idx], dtype=torch.float32),
                torch.tensor(self.idiq_next[idx], dtype=torch.float32),
                torch.tensor(self.we[idx], dtype=torch.float32))

def create_dataloader(folder_names, batch_size, shuffle=True, num_workers=0):
    dataset = MotorDataset(folder_names)
    if len(dataset) == 0:
        raise ValueError("[Error] Dataset is empty, please check the path configuration.")
    
    # shuffle=True is safe here because the differential is already computed inside the Dataset
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, 
                            drop_last=True, num_workers=num_workers)
    return dataloader
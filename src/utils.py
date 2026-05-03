import os
import random
import yaml
import torch
import numpy as np
import pandas as pd

def set_seed(seed=42):
    """Fix random seed to ensure experiment reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Usually, deterministic is not enabled for performance reasons unless strict reproducibility is required.
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False
    print(f"[Utils] Random seed set to {seed}")

def load_config(config_path):
    """Load YAML configuration file."""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

def get_device():
    """Get computation device."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Utils] Using device: {device}")
    return device

class ExperimentLogger:
    """
    Experiment Logger
    Responsible for collecting loss and metrics during training and saving them to CSV/Excel.
    """
    def __init__(self, save_dir, filename="training_log.xlsx"):
        self.save_path = os.path.join(save_dir, filename)
        os.makedirs(save_dir, exist_ok=True)
        self.history = {}

    def log_metrics(self, epoch, metrics_dict):
        """
        Log metrics for one Epoch
        metrics_dict: {'loss': 0.1, 'mse': 0.05, 'lr': 0.001, ...}
        """
        if 'epoch' not in self.history:
            self.history['epoch'] = []
        self.history['epoch'].append(epoch)

        for k, v in metrics_dict.items():
            if k not in self.history:
                self.history[k] = []
            self.history[k].append(v)

    def save(self):
        """Save to disk."""
        df = pd.DataFrame(self.history)
        if self.save_path.endswith('.xlsx'):
            df.to_excel(self.save_path, index=False)
        else:
            df.to_csv(self.save_path, index=False)
        # print(f"[Logger] Log saved to {self.save_path}")
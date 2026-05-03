import torch
import torch.nn as nn
import numpy as np
import copy
import random

class HierarchicalAdaptiveLossWeights(nn.Module):
    """
    Hierarchical Adaptive Weighting (Macro-Micro)
    [Macro]: The total physical contribution is strictly anchored to a target_phys_ratio of MSE (e.g., 20%),
             ensuring the absolute dominance of MSE.
    [Micro]: The physical group internally restores uncertainty weighting (log_sigma), where physical losses 
             with higher difficulty/noise automatically receive a smaller weight allocation.
    """
    def __init__(self, n_losses, target_phys_ratio=0.2, alpha=0.9, max_micro_weight=100.0):
        """
        Args:
            n_losses: Total number of loss terms (currently 4: [MSE, PDE1, PDE2, Sym])
            target_phys_ratio: Proportion of total physical constraints relative to MSE (default 0.2)
            alpha: Macro EMA smoothing coefficient to prevent oscillation (default 0.9)
            max_micro_weight: Micro weight upper bound to block the infinite negative reward vulnerability of log(sigma) (default 100.0)
        """
        super().__init__()
        self.n_phys = n_losses - 1
        self.target_ratio = target_phys_ratio
        self.alpha = alpha

        # 1. Micro level: Uncertainty parameters (for weight allocation within the physics group)
        # Calculate the lower bound to block the vulnerability where the network gets infinite negative return by making sigma->0
        self.min_log_sigma = -np.log(max_micro_weight)
        self.log_sigmas = nn.Parameter(torch.zeros(self.n_phys))

        # 2. Macro level: EMA smoothing tracker (does not participate in gradients)
        self.register_buffer('ema_mse', torch.tensor(1.0))
        # Initialize ema_phys_eff as the number of physical terms, because the optimal micro effective contribution (W_micro * L) theoretically approaches 1
        self.register_buffer('ema_phys_eff', torch.tensor(float(self.n_phys))) 

        # Record the final applied physical weights (only for logging)
        self.current_final_weights = [0.0] * self.n_phys

    def forward(self, losses):
        mse_loss = losses[0]
        phys_losses = torch.stack(losses[1:])

        # ================= 1. Micro Allocation (Uncertainty Weighting) =================
        clamped_log_sigmas = torch.clamp(self.log_sigmas, min=self.min_log_sigma)
        micro_weights = torch.exp(-clamped_log_sigmas)

        # Combined Loss within the physics group (with log penalty term)
        phys_group_loss = torch.sum(micro_weights * phys_losses + clamped_log_sigmas)

        # ================= 2. Macro Control (Dynamic Scaling Coefficient M) =================
        if self.training:
            with torch.no_grad(): # Detach graph, macro scaling must absolutely not interfere with the main gradient
                # Calculate the internal "total effective contribution" of the current physics group
                current_phys_eff = torch.sum(micro_weights * phys_losses)

                # Update EMA
                self.ema_mse = self.alpha * self.ema_mse + (1 - self.alpha) * mse_loss.detach()
                self.ema_phys_eff = self.alpha * self.ema_phys_eff + (1 - self.alpha) * current_phys_eff

                # Calculate macro scaling coefficient M: such that M * EMA_phys_eff = Target_Ratio * EMA_MSE
                macro_M = (self.target_ratio * self.ema_mse) / (self.ema_phys_eff + 1e-8)
                
                # Record the final applied weights (for print logs): W_final = M * W_micro
                self.current_final_weights = (macro_M * micro_weights).cpu().numpy().tolist()
        else:
            # In Eval mode, directly use the long-term EMA mean to calculate M
            macro_M = (self.target_ratio * self.ema_mse) / (self.ema_phys_eff + 1e-8)

        # ================= 3. Final Assembly =================
        # Total = 1.0 * MSE + M * (Micro uncertainty physical loss)
        # Multiplying M externally perfectly preserves the optimal derivative solution of internal sigma while capping the total upper limit
        total_loss = mse_loss + macro_M * phys_group_loss

        return total_loss

    def get_weights(self):
        """Provide for log printing: return the final superimposed [w_mse, w_pde1, w_pde2, w_sym]"""
        return [1.0] + self.current_final_weights 
    

class StaticLossWeights(nn.Module):
    """
    Static weights for use with PCGrad.
    Since PCGrad resolves gradient direction conflicts from the bottom up, we no longer need adaptive weights that easily lead to crashes.
    Just give MSE an absolute order of magnitude advantage.
    """
    def __init__(self, w_mse=1.0, w_phys=0.1):
        super().__init__()
        self.w_mse = w_mse
        self.w_phys = w_phys

    def forward(self, losses):
        """Note: This returns a weighted [list] because PCGrad needs to calculate gradients task by task"""
        weighted_losses = [
            losses[0] * self.w_mse,       # MSE
            losses[1] * 0,                # PDE1 (Disabled due to noise sensitivity)
            losses[2] * 0,                # PDE2 (Disabled due to noise sensitivity)
            losses[3] * self.w_phys       # SYM
        ]
        return weighted_losses

    def get_weights(self):
        """Interface for log printing"""
        return [self.w_mse, self.w_phys, self.w_phys, self.w_phys]


class PCGrad:
    """
    PCGrad (Projected Gradient Descent) Optimizer Wrapper Class
    Reference: Gradient Surgery for Multi-Task Learning (NeurIPS 2020)
    """
    def __init__(self, optimizer):
        self.optimizer = optimizer
    
    # Allow the wrapper to pass the param_groups attribute to the internal native optimizer
    @property
    def param_groups(self):
        return self.optimizer.param_groups

    def zero_grad(self):
        self.optimizer.zero_grad(set_to_none=True)

    def step(self):
        self.optimizer.step()

    def pc_backward(self, objectives):
        """
        Core method: Calculate projected gradients and set them in the network parameters
        Args:
            objectives: list of loss tensors (e.g., [loss_mse, loss_pde1, ...])
        """
        grads, params = [], []
        
        # Collect all parameters requiring gradients
        for group in self.optimizer.param_groups:
            for p in group['params']:
                if p.requires_grad:
                    params.append(p)

        # 1. Calculate gradients independently for each task
        for obj in objectives:
            self.optimizer.zero_grad(set_to_none=True)
            # Retain computational graph because we need to backward repeatedly for multiple losses
            obj.backward(retain_graph=True) 
            
            grad_vec = []
            for p in params:
                if p.grad is not None:
                    grad_vec.append(p.grad.data.clone().flatten())
                else:
                    grad_vec.append(torch.zeros_like(p.data).flatten())
            # Assemble into a super-long 1D vector
            grads.append(torch.cat(grad_vec))

        # 2. Gradient projection surgery (eliminate conflicts)
        pc_grads = copy.deepcopy(grads)
        for g_i in pc_grads:
            # Randomly shuffle comparison order to improve generalization
            random.shuffle(grads)
            for g_j in grads:
                dot_product = torch.dot(g_i, g_j)
                # If dot product < 0, a conflict exists! Perform orthogonal projection to remove conflicting components
                if dot_product < 0:
                    g_i -= (dot_product / (torch.dot(g_j, g_j) + 1e-8)) * g_j

        # 3. Accumulate the safe gradients after "surgery"
        merged_grad = torch.stack(pc_grads).sum(dim=0)

        # 4. Restore back into the .grad attribute of network parameters
        self.optimizer.zero_grad(set_to_none=True)
        idx = 0
        for p in params:
            length = p.numel()
            p.grad = merged_grad[idx : idx + length].view_as(p).clone()
            idx += length

class ProportionalAdaptiveLossWeights(nn.Module):
    """
    Proportional Adaptive Multi-Task Loss Weighting
    Logic: Physical constraints use uncertainty adaptive weighting. The MSE weight is dynamically adjusted
    so that the effective loss contribution of MSE (W_mse * L_mse) always accounts for a fixed proportion (e.g., 80%) of the total effective loss contribution.
    """
    def __init__(self, n_losses, mse_ratio=0.8, max_phys_weight=1.0):
        """
        Args:
            n_losses (int): Total number of loss terms (currently 4: [MSE, PDE1, PDE2, Sym])
            mse_ratio (float): Target proportion of MSE effective contribution to total contribution (default 0.8, i.e., 80%)
            max_phys_weight (float): Maximum absolute weight upper bound allowed for physical constraints
        """
        super().__init__()
        self.n_phys_losses = n_losses - 1
        self.mse_ratio = mse_ratio
        
        # Calculate the truncated lower bound of physical weights
        self.min_log_sigma = -torch.log(torch.tensor(max_phys_weight))
        self.log_sigmas = nn.Parameter(torch.zeros(self.n_phys_losses))
        
        # Record the current dynamic weight of MSE for log printing and plotting
        self.current_w_mse = 1.0

    def forward(self, losses):
        """
        Args:
            losses (list of tensors): [loss_mse, loss_pde1, loss_pde2, loss_sym]
        """
        mse_loss = losses[0]
        
        phys_loss_total = 0.0      # Total physical loss containing gradients (for backpropagation)
        eff_phys_loss_val = 0.0    # True effective contribution sum of physical terms (for numerical calculation only, no gradients)
        
        # 1. Calculate effective loss and adaptive weights for physical constraints
        for i in range(self.n_phys_losses):
            clamped_log_sigma = torch.clamp(self.log_sigmas[i], min=self.min_log_sigma.item())
            precision = torch.exp(-clamped_log_sigma)
            
            # Accumulate effective physical contribution (Crucial: MUST detach gradient, otherwise it destroys MSE optimization direction!)
            eff_phys_loss_val += (precision * losses[i + 1]).detach()
            
            # Accumulate physical loss for backpropagation (includes log_sigma penalty term)
            phys_loss_total += precision * losses[i + 1] + clamped_log_sigma
            
        # 2. Dynamically calculate the current weight of MSE based on target proportion
        # W_mse = (r / (1 - r)) * E_phys / L_mse
        ratio_factor = self.mse_ratio / (1.0 - self.mse_ratio)
        
        # Prevent L_mse approaching 0 causing division by zero error
        w_mse = ratio_factor * eff_phys_loss_val / (mse_loss.detach() + 1e-8)
        
        # Add reasonable numerical truncation to prevent weight collapse into astronomical figures due to abnormal early batches
        w_mse = torch.clamp(w_mse, min=1e-3, max=1e4)
        
        self.current_w_mse = w_mse.item()
        
        # 3. Combine final total loss (MSE multiplied by w_mse as a constant)
        total_loss = w_mse * mse_loss + phys_loss_total
        
        return total_loss

    def get_weights(self):
        """Get the current visualizable weight value list [w_mse, w_pde1, w_pde2, w_sym]"""
        with torch.no_grad():
            ws = [self.current_w_mse]
            clamped_sigmas = torch.clamp(self.log_sigmas, min=self.min_log_sigma.item())
            phys_ws = torch.exp(-clamped_sigmas).cpu().numpy().tolist()
            ws.extend(phys_ws)
            return ws
        

class CurriculumLossWeights(nn.Module):
    """
    Curriculum Learning (Warm-up) Multi-Task Weighting
    Early stage purely fits data (MSE), later stage slowly introduces physical constraints (PDE/Sym).
    """
    def __init__(self, max_epochs, warm_up_ratio=0.2, max_phys_weight=1e-10):
        """
        Args:
            max_epochs: Total training epochs
            warm_up_ratio: Warm-up period proportion (default 0.2, meaning the first 20% of time only fits MSE)
            max_phys_weight: Final maximum weight for physical constraints (default 0.05)
        """
        super().__init__()
        self.max_epochs = max_epochs
        self.warm_up_epoch = int(max_epochs * warm_up_ratio) 
        self.max_phys_weight = max_phys_weight
        
        self.current_epoch = 1
        self.current_phys_w = 0.0

    def step_epoch(self, epoch):
        """Passed from external train.py every epoch to update the physical weight for the current stage"""
        self.current_epoch = epoch
        
        # Warm-up period: physical constraint weights are strictly 0
        if epoch <= self.warm_up_epoch:
            self.current_phys_w = 0.0
        else:
            # After warm-up, use Cosine curve to smoothly transition to maximum weight
            progress = (epoch - self.warm_up_epoch) / (self.max_epochs - self.warm_up_epoch)
            factor = 0.5 * (1 - np.cos(np.pi * progress))
            self.current_phys_w = self.max_phys_weight * factor

    def forward(self, losses):
        """
        Args:
            losses (list of tensors): [loss_mse, loss_pde1, loss_pde2, loss_sym]
        """
        # MSE weight is always fixed at 1.0
        total_loss = losses[0] * 1.0 
        
        # Uniformly multiply the physical weight of the current stage onto PDE and Sym
        for i in range(1, len(losses)):
            total_loss += self.current_phys_w * losses[i]
            
        return total_loss

    def get_weights_dict(self):
        """Return dictionary format to interface with train_engine.py logging"""
        return {
            'w_mse': 1.0,
            'w_pde1': float(self.current_phys_w),
            'w_pde2': float(self.current_phys_w),
            'w_sym': float(self.current_phys_w)
        }
    

class AnchoredAdaptiveLossWeights(nn.Module):
    """
    Anchored Adaptive Multi-Task Loss Weighting
    Ensures MSE is anchored at 1.0, applying restricted adaptive weighting only to physical constraints.
    """
    def __init__(self, n_losses, max_phys_weight=1.0):
        """
        Args:
            n_losses (int): Total number of loss terms (current project is 4: [MSE, PDE1, PDE2, Sym])
            max_phys_weight (float): Maximum absolute weight upper bound allowed for physical constraints
        """
        super().__init__()
        # Apply adaptive weighting only to the remaining (n_losses - 1) physical losses
        self.n_phys_losses = n_losses - 1
        self.max_phys_weight = max_phys_weight
        
        # Calculate truncated lower bound: exp(-min_log_sigma) = max_phys_weight
        self.min_log_sigma = -torch.log(torch.tensor(max_phys_weight))
        
        # Currently log_sigmas only has 3 parameters (corresponding to PDE1, PDE2, Sym)
        self.log_sigmas = nn.Parameter(torch.zeros(self.n_phys_losses))

    def forward(self, losses):
        """
        Args:
            losses (list of tensors): [loss_mse, loss_pde1, loss_pde2, loss_sym]
        """
        # 1. Anchor MSE loss (always core dominant, weight is 1.0)
        mse_loss = losses[0]
        total_loss = mse_loss * 1.0
        
        # 2. Apply restricted adaptive weighting to physical constraints
        for i in range(self.n_phys_losses):
            # Limit log_sigma from shrinking infinitely, thereby preventing physical weights from exploding
            clamped_log_sigma = torch.clamp(self.log_sigmas[i], min=self.min_log_sigma.item())
            
            precision = torch.exp(-clamped_log_sigma)
            total_loss += precision * losses[i + 1] + clamped_log_sigma
            
        return total_loss

    def get_weights(self):
        """Get the current visualizable weight value list"""
        with torch.no_grad():
            # MSE weight is always 1.0
            ws = [1.0]
            # Calculate truncated physical weights and append to list
            clamped_sigmas = torch.clamp(self.log_sigmas, min=self.min_log_sigma.item())
            phys_ws = torch.exp(-clamped_sigmas).cpu().numpy().tolist()
            ws.extend(phys_ws)
            return ws
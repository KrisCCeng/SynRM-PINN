import torch
import torch.nn as nn

# ==========================================
# 1. Traditional Black-box Neural Network (Traditional ANN) - No structural constraints
# ==========================================
class TraditionalANN(nn.Module):
    def __init__(self, i_norm=4.0, psi_norm=1.5, hidden_dim=(6, 4)):
        super().__init__()
        self.mode = "ANN"
        
        # Normalization parameters (Buffer: no gradient update)
        self.register_buffer('i_scale', torch.tensor(i_norm, dtype=torch.float32))
        self.register_buffer('psi_scale', torch.tensor(psi_norm, dtype=torch.float32))

        if isinstance(hidden_dim, int):
            h1, h2 = hidden_dim, hidden_dim
        elif isinstance(hidden_dim, (list, tuple)) and len(hidden_dim) == 2:
            h1, h2 = hidden_dim[0], hidden_dim[1]
        else:
            raise ValueError("hidden_dim must be a single integer or a list/tuple containing two integers")

        # Traditional black-box structure: directly fit current to flux mapping, no implicit inductance, no absolute symmetry
        self.net = nn.Sequential(
            nn.Linear(2, h1),
            nn.ELU(),
            nn.Linear(h1, h2),
            nn.ELU(),
            nn.Linear(h2, 2)  # Directly output Psi_d, Psi_q
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward_flux(self, id_iq_phys):
        """Direct forward propagation to compute flux linkage"""
        # 1. Normalized input (no absolute value, lacks parity symmetry)
        i_norm = id_iq_phys / self.i_scale
        
        # 2. Network directly outputs normalized flux predictions (cannot guarantee passing origin at zero current)
        psi_norm_pred = self.net(i_norm)
        
        # 3. Restore physical scale
        psid_pred = psi_norm_pred[:, 0] * self.psi_scale
        psiq_pred = psi_norm_pred[:, 1] * self.psi_scale
        
        return torch.stack([psid_pred, psiq_pred], dim=1)

    def forward(self, inputs, id_iq_now, id_iq_next, we):
        """Unified interface: Traditional ANN only needs current id_iq_now for fitting"""
        return self.forward_flux(id_iq_now)


# ==========================================
# 2. Structure-Constrained Network & Physics-Informed Neural Network (SCN & PINN)
# ==========================================
class SymmetricSynRMModel(nn.Module):
    def __init__(self, dt=1e-4, mode="PINN", i_norm=4.0, psi_norm=1.5, hidden_dim=(6, 4)):
        """
        Symmetric Synchronous Reluctance Motor Model
        Args:
            mode: "SCN" (Structure constraint only) or "PINN" (Full constraints)
        """
        super().__init__()
        self.R = 10.281455573854170  
        self.dt = dt
        self.mode = mode # Currently "SCN" or "PINN"

        self.register_buffer('i_scale', torch.tensor(i_norm, dtype=torch.float32))
        self.register_buffer('psi_scale', torch.tensor(psi_norm, dtype=torch.float32))
        self.L_norm_factor = psi_norm / i_norm

        if isinstance(hidden_dim, int):
            h1, h2 = hidden_dim, hidden_dim
        elif isinstance(hidden_dim, (list, tuple)) and len(hidden_dim) == 2:
            h1, h2 = hidden_dim[0], hidden_dim[1]

        self.net = nn.Sequential(
            nn.Linear(2, h1),
            nn.ELU(),
            nn.Linear(h1, h2),
            nn.ELU(),
            nn.Linear(h2, 2)
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward_flux(self, id_iq_phys):
        """Implicitly learn secant inductance (satisfies hard constraints)"""
        i_norm = id_iq_phys / self.i_scale
        i_abs = torch.abs(i_norm) # Even symmetry hard constraint
        
        L_pred_norm = self.net(i_abs) 
        
        L_d_val = L_pred_norm[:, 0]
        L_q_val = L_pred_norm[:, 1]
        
        # Implicit inductance multiplication (guarantees zero-point constraint)
        psid_pred = L_d_val * id_iq_phys[:, 0] * self.L_norm_factor
        psiq_pred = L_q_val * id_iq_phys[:, 1] * self.L_norm_factor
        
        return torch.stack([psid_pred, psiq_pred], dim=1)

    def forward(self, inputs, id_iq_now, id_iq_next, we):
        if self.mode == "PINN":
            id_iq_now.requires_grad_(True)
        
        flux_pred = self.forward_flux(id_iq_now)

        # SCN mode relies only on the above structural constraints, directly returns flux, and does not compute physical partial derivative residuals
        if self.mode == "SCN":
            return flux_pred

        # --- PINN physical derivative calculation ---
        psid, psiq = flux_pred[:, 0], flux_pred[:, 1]
        
        grads_d = torch.autograd.grad(psid.sum(), id_iq_now, create_graph=True)[0]
        grads_q = torch.autograd.grad(psiq.sum(), id_iq_now, create_graph=True)[0]
        
        Ldd, Ldq = grads_d[:, 0], grads_d[:, 1]
        Lqd, Lqq = grads_q[:, 0], grads_q[:, 1]
        
        delta_id = id_iq_next[:, 0] - id_iq_now[:, 0]
        delta_iq = id_iq_next[:, 1] - id_iq_now[:, 1]
        
        ud, uq = inputs[:, 0], inputs[:, 1]
        omega = we[:, 0]
        
        delta_psid = Ldd * delta_id + Ldq * delta_iq
        delta_psiq = Lqd * delta_id + Lqq * delta_iq
        
        res_d_scaled = delta_psid + (self.R * id_iq_now[:, 0] - omega * psiq - ud) * self.dt
        res_q_scaled = delta_psiq + (self.R * id_iq_now[:, 1] + omega * psid - uq) * self.dt
        
        loss_pde1 = torch.mean(res_d_scaled ** 2)
        loss_pde2 = torch.mean(res_q_scaled ** 2)
        loss_sym = torch.mean((Ldq - Lqd) ** 2)
        
        return flux_pred, loss_pde1, loss_pde2, loss_sym
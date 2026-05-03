# File path: src/baselines.py

import numpy as np
import torch
import torch.nn as nn
from scipy.interpolate import RegularGridInterpolator

# ==========================================
# 1. Look-Up Table (LUT) Model
# ==========================================
class LookUpTableModel:
    def __init__(self):
        # Physical constants (from evaluation.py)
        self.I_RATED = 2.1072
        self.STEP_PU = 0.1
        self.STEP_AMP = self.STEP_PU * self.I_RATED
        
        # Initialize table data
        self._init_tables()
        
        # Build grid coordinates (Row: Iq, Col: Id)
        rows, cols = self.PsidTable.shape
        self.iq_grid = np.arange(rows) * self.STEP_AMP
        self.id_grid = np.arange(cols) * self.STEP_AMP
        
        # Create interpolators (Note the scaling factor 10000)
        # RegularGridInterpolator ((x_axis, y_axis), data)
        # Here x_axis corresponds to Iq (Rows), y_axis corresponds to Id (Cols)
        self.interp_d = RegularGridInterpolator(
            (self.iq_grid, self.id_grid), 
            self.PsidTable / 10000.0, 
            bounds_error=False, 
            fill_value=None
        )
        self.interp_q = RegularGridInterpolator(
            (self.iq_grid, self.id_grid), 
            self.PsiqTable / 10000.0, 
            bounds_error=False, 
            fill_value=None
        )

    def _init_tables(self):
        # 请务必在此处填入完整的 numpy array 数据！
        self.PsidTable = np.array([
            [0, 1940, 3816, 5572, 7165, 8558, 9726, 10665, 11399, 11975, 12440, 12826, 13151, 13425, 13661, 13866, 14045, 14202, 14340, 14456, 14531],
            [0, 1935, 3806, 5558, 7148, 8538, 9706, 10645, 11380, 11957, 12423, 12810, 13134, 13410, 13647, 13853, 14035, 14195, 14336, 14455, 14531],
            [0, 1923, 3782, 5523, 7103, 8487, 9652, 10592, 11330, 11911, 12379, 12767, 13094, 13372, 13613, 13823, 14011, 14178, 14326, 14452, 14532],
            [0, 1908, 3751, 5477, 7044, 8420, 9581, 10522, 11263, 11848, 12320, 12712, 13044, 13328, 13573, 13788, 13983, 14158, 14314, 14447, 14531],
            [0, 1892, 3718, 5428, 6982, 8349, 9506, 10446, 11190, 11778, 12255, 12653, 12992, 13284, 13535, 13756, 13956, 14139, 14301, 14438, 14525],
            [0, 1875, 3685, 5379, 6920, 8278, 9430, 10370, 11116, 11708, 12188, 12592, 12938, 13239, 13497, 13723, 13929, 14116, 14284, 14424, 14514],
            [0, 1858, 3651, 5330, 6859, 8208, 9356, 10296, 11044, 11638, 12122, 12529, 12881, 13190, 13456, 13688, 13898, 14089, 14261, 14406, 14499],
            [0, 1842, 3619, 5283, 6800, 8141, 9284, 10222, 10973, 11571, 12057, 12466, 12823, 13138, 13410, 13648, 13862, 14057, 14234, 14384, 14481],
            [0, 1826, 3588, 5238, 6742, 8074, 9211, 10149, 10901, 11503, 11993, 12406, 12764, 13081, 13358, 13601, 13820, 14021, 14204, 14360, 14462],
            [0, 1811, 3557, 5192, 6682, 8003, 9134, 10072, 10828, 11433, 11928, 12345, 12705, 13021, 13299, 13547, 13774, 13982, 14172, 14334, 14440],
            [0, 1796, 3525, 5140, 6614, 7920, 9045, 9985, 10747, 11359, 11860, 12283, 12645, 12961, 13241, 13493, 13726, 13941, 14135, 14302, 14411],
            [0, 1781, 3490, 5082, 6533, 7823, 8942, 9887, 10659, 11279, 11789, 12219, 12585, 12902, 13186, 13444, 13680, 13898, 14095, 14266, 14378],
            [0, 1766, 3454, 5021, 6447, 7722, 8837, 9788, 10569, 11198, 11716, 12153, 12522, 12845, 13134, 13396, 13634, 13853, 14054, 14229, 14344],
            [0, 1752, 3421, 4962, 6364, 7625, 8739, 9693, 10480, 11117, 11641, 12079, 12455, 12785, 13079, 13343, 13583, 13805, 14011, 14189, 14307],
            [0, 1737, 3389, 4908, 6289, 7538, 8646, 9600, 10391, 11036, 11563, 12004, 12386, 12721, 13017, 13282, 13525, 13752, 13964, 14149, 14271],
            [0, 1722, 3357, 4858, 6222, 7458, 8558, 9506, 10299, 10952, 11490, 11939, 12324, 12659, 12951, 13217, 13467, 13702, 13919, 14111, 14239],
            [0, 1707, 3325, 4808, 6156, 7379, 8468, 9408, 10201, 10865, 11416, 11876, 12265, 12599, 12891, 13160, 13415, 13655, 13874, 14068, 14199],
            [0, 1693, 3293, 4757, 6088, 7297, 8376, 9310, 10105, 10773, 11330, 11794, 12189, 12533, 12835, 13109, 13365, 13603, 13819, 14010, 14139],
            [0, 1678, 3260, 4704, 6016, 7211, 8284, 9220, 10017, 10685, 11240, 11704, 12102, 12456, 12771, 13052, 13306, 13543, 13761, 13950, 14076],
            [0, 1663, 3227, 4648, 5939, 7119, 8188, 9131, 9938, 10607, 11157, 11619, 12022, 12385, 12708, 12994, 13250, 13489, 13712, 13904, 14030],
            [0, 1652, 3202, 4606, 5879, 7046, 8112, 9064, 9881, 10552, 11096, 11554, 11964, 12335, 12663, 12951, 13209, 13451, 13681, 13878, 14006]
        ])
        
        self.PsiqTable = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [771, 766, 753, 731, 705, 675, 645, 614, 584, 558, 534, 513, 494, 476, 459, 444, 430, 418, 406, 396, 389],
            [1393, 1386, 1366, 1334, 1293, 1247, 1198, 1148, 1100, 1056, 1015, 976, 940, 908, 879, 853, 828, 805, 784, 766, 753],
            [1846, 1839, 1818, 1785, 1742, 1691, 1637, 1580, 1524, 1470, 1418, 1367, 1319, 1279, 1243, 1211, 1180, 1149, 1121, 1097, 1081],
            [2194, 2188, 2168, 2136, 2093, 2043, 1987, 1928, 1870, 1812, 1752, 1692, 1637, 1593, 1556, 1522, 1487, 1451, 1416, 1388, 1370],
            [2491, 2484, 2464, 2433, 2390, 2340, 2283, 2224, 2165, 2104, 2038, 1969, 1910, 1866, 1831, 1796, 1759, 1719, 1679, 1647, 1628],
            [2756, 2749, 2730, 2698, 2655, 2604, 2546, 2486, 2425, 2361, 2290, 2215, 2153, 2112, 2081, 2046, 2005, 1960, 1917, 1882, 1862],
            [2999, 2992, 2973, 2941, 2898, 2845, 2787, 2725, 2662, 2596, 2522, 2442, 2377, 2338, 2310, 2276, 2230, 2179, 2133, 2098, 2078],
            [3225, 3219, 3199, 3167, 3123, 3070, 3009, 2946, 2881, 2814, 2737, 2655, 2588, 2547, 2519, 2483, 2434, 2380, 2334, 2302, 2282],
            [3441, 3434, 3413, 3379, 3333, 3279, 3217, 3151, 3084, 3014, 2937, 2856, 2786, 2740, 2708, 2671, 2623, 2570, 2526, 2496, 2477],
            [3649, 3641, 3616, 3578, 3529, 3472, 3408, 3340, 3268, 3193, 3118, 3041, 2970, 2918, 2885, 2849, 2803, 2753, 2709, 2678, 2659],
            [3851, 3841, 3810, 3765, 3712, 3652, 3585, 3514, 3435, 3355, 3282, 3213, 3144, 3090, 3057, 3023, 2977, 2926, 2880, 2848, 2829],
            [4041, 4028, 3992, 3940, 3883, 3821, 3754, 3680, 3596, 3512, 3442, 3377, 3313, 3262, 3226, 3191, 3143, 3089, 3043, 3009, 2989],
            [4218, 4204, 4162, 4104, 4043, 3981, 3914, 3838, 3755, 3674, 3603, 3535, 3476, 3427, 3386, 3345, 3298, 3248, 3201, 3163, 3138],
            [4385, 4370, 4326, 4263, 4197, 4134, 4065, 3986, 3908, 3838, 3767, 3694, 3633, 3582, 3532, 3485, 3441, 3399, 3355, 3313, 3285],
            [4544, 4529, 4486, 4421, 4352, 4287, 4213, 4131, 4057, 3995, 3930, 3858, 3793, 3731, 3669, 3616, 3574, 3537, 3501, 3464, 3437],
            [4695, 4682, 4640, 4577, 4510, 4444, 4369, 4284, 4207, 4143, 4080, 4015, 3949, 3881, 3818, 3765, 3718, 3677, 3645, 3615, 3590],
            [4841, 4828, 4788, 4726, 4659, 4590, 4514, 4433, 4357, 4286, 4221, 4159, 4093, 4027, 3969, 3918, 3868, 3822, 3790, 3766, 3745],
            [4980, 4967, 4928, 4864, 4790, 4714, 4639, 4566, 4495, 4426, 4361, 4297, 4228, 4160, 4100, 4048, 4001, 3960, 3930, 3910, 3899],
            [5105, 5093, 5052, 4986, 4906, 4828, 4756, 4688, 4621, 4553, 4486, 4419, 4352, 4283, 4216, 4160, 4117, 4086, 4063, 4045, 4034],
            [5191, 5178, 5138, 5070, 4989, 4914, 4846, 4780, 4711, 4642, 4569, 4495, 4428, 4362, 4291, 4229, 4187, 4164, 4154, 4141, 4128]
        ])

# Safety check: prevent crashes if data is missing
        if self.PsidTable.shape[0] < 5:
            print("[Warning] Baselines: Table data incomplete. Please provide full arrays in src/baselines.py!")
            # Temporary expansion to prevent crash
            self.PsidTable = np.zeros((21, 21))
            self.PsiqTable = np.zeros((21, 21))

    def predict(self, id_val, iq_val):
        """
        LUT prediction (supports scalar or numpy array)
        Symmetry handling:
        Psi_d(id, iq) is an odd function w.r.t id, even function w.r.t iq
        Psi_q(id, iq) is an even function w.r.t id, odd function w.r.t iq
        """
        # 1. Map to the first quadrant (Table only covers the 1st quadrant)
        id_abs = np.abs(id_val)
        iq_abs = np.abs(iq_val)
        
        # 2. Construct query points [Iq, Id] (Note the order matches RegularGridInterpolator initialization)
        points = np.stack([iq_abs, id_abs], axis=-1)
        
        # 3. Interpolation
        psid_abs = self.interp_d(points)
        psiq_abs = self.interp_q(points)
        
        # 4. Restore signs (Symmetry)
        # Psi_d: odd w.r.t id (sign(id)), even w.r.t iq
        psid = np.sign(id_val) * psid_abs
        
        # Psi_q: odd w.r.t iq (sign(iq)), even w.r.t id
        psiq = np.sign(iq_val) * psiq_abs
        
        return psid, psiq

# ==========================================
# 2. Polynomial Model (Numerical Inverse - Batch Optimization)
# ==========================================
class PolynomialModel:
    def __init__(self):
        # Default physical coefficients (used for simulation or as initial values before fitting)
        self.ad0 = 0.992485762
        self.add = 0.266009212
        self.aq0 = 3.00064969
        self.aqq = 8.62481689
        self.adq = 2.8628087

    def fit(self, data_dict):
        """
        Linear Least Squares (LLS) fitting using experimental data, 
        strictly following IEEE paper Eq. 15 ~ Eq. 22.
        """
        # 1. D-axis self-saturation fitting (Eq. 15 - Eq. 18)
        if 'd_axis' in data_dict:
            id_d, iq_d, pd_d, pq_d = data_dict['d_axis']
            valid = np.abs(pd_d) > 0.05  # Filter out zero region to avoid division by zero
            if np.any(valid):
                y = id_d[valid] / pd_d[valid]
                # X matrix: [1, |psi_d|^5]
                X = np.stack([np.ones_like(y), np.abs(pd_d[valid])**5], axis=1)
                beta_d = np.linalg.lstsq(X, y, rcond=None)[0]
                self.ad0, self.add = beta_d[0], beta_d[1]

        # 2. Q-axis self-saturation fitting
        if 'q_axis' in data_dict:
            id_q, iq_q, pd_q, pq_q = data_dict['q_axis']
            valid = np.abs(pq_q) > 0.05
            if np.any(valid):
                y = iq_q[valid] / pq_q[valid]
                # X matrix: [1, |psi_q|^1]
                X = np.stack([np.ones_like(y), np.abs(pq_q[valid])**1], axis=1)
                beta_q = np.linalg.lstsq(X, y, rcond=None)[0]
                self.aq0, self.aqq = beta_q[0], beta_q[1]

        # 3. Cross-saturation fitting (Eq. 19 - Eq. 22)
        if 'cross' in data_dict:
            id_c, iq_c, pd_c, pq_c = data_dict['cross']
            
            # Construct target vector y_dq
            y1 = id_c - self.ad0 * pd_c - self.add * (np.abs(pd_c)**5) * pd_c
            y2 = iq_c - self.aq0 * pq_c - self.aqq * (np.abs(pq_c)**1) * pq_c
            y_cross = np.concatenate([y1, y2])
            
            # Construct regression matrix X_dq
            x1 = 0.5 * np.abs(pd_c)**1 * np.abs(pq_c)**2 * pd_c
            x2 = 0.5 * np.abs(pd_c)**2 * np.abs(pq_c)**0 * pq_c
            X_cross = np.concatenate([x1, x2])[:, np.newaxis]
            
            beta_cross = np.linalg.lstsq(X_cross, y_cross, rcond=None)[0]
            self.adq = beta_cross[0]
            
        print(f"  [Polynomial Fitted] ad0={self.ad0:.4f}, add={self.add:.4f}, aq0={self.aq0:.4f}, aqq={self.aqq:.4f}, adq={self.adq:.4f}")

    def _id_formula(self, phid, phiq):
        term1 = self.ad0
        term2 = self.add * torch.abs(phid)**5
        term3 = (self.adq / 2) * torch.abs(phid)**1 * torch.abs(phiq)**2
        return (term1 + term2 + term3) * phid

    def _iq_formula(self, phid, phiq):
        term1 = self.aq0
        term2 = self.aqq * torch.abs(phiq)**1
        term3 = (self.adq / 2) * torch.abs(phid)**2 * torch.abs(phiq)**0
        return (term1 + term2 + term3) * phiq

    def numerical_inverse(self, target_id_np, target_iq_np, max_iter=1000, lr=0.1):
        """Batch numerical inverse: Given id, iq, solve for psid, psiq"""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        t_id = torch.tensor(target_id_np, dtype=torch.float32).view(-1).to(device)
        t_iq = torch.tensor(target_iq_np, dtype=torch.float32).view(-1).to(device)
        
        psi_d = (t_id * 0.5).clone().detach().requires_grad_(True)
        psi_q = (t_iq * 0.2).clone().detach().requires_grad_(True)
        
        optimizer = torch.optim.Adam([psi_d, psi_q], lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=50)

        for i in range(max_iter):
            optimizer.zero_grad()
            i_d_pred = self._id_formula(psi_d, psi_q)
            i_q_pred = self._iq_formula(psi_d, psi_q)
            loss = torch.mean((i_d_pred - t_id)**2 + (i_q_pred - t_iq)**2)
            loss.backward()
            optimizer.step()
            scheduler.step(loss)
            if loss.item() < 1e-7:
                break
        
        return psi_d.detach().cpu().numpy(), psi_q.detach().cpu().numpy()
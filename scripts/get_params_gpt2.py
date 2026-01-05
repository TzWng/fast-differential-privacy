import torch
import math

# ==============================================================================
# 1. Helper Function: Spectral Metric for Raw Noise (Clipping/Noise Calculation)
# ==============================================================================
def get_raw_spectral_metric(shape):
    """
    Calculates the Spectral Norm of the RAW Gaussian Noise matrix.
    Used for: _get_noise4target, _get_clip4target
    """
    # === Case A: Linear Layer (Matrix) [Out, In] ===
    # Theory: Spectral norm ~ sqrt(N) + sqrt(M) (Power Law)
    if len(shape) == 2:
        n_out, n_in = shape
        return n_out**0.5 + n_in**0.5
    
    # === Case B: Conv2d (4D Tensor) ===
    elif len(shape) == 4:
        n_out = shape[0]
        n_in = shape[1] * shape[2] * shape[3]
        return n_out**0.5 + n_in**0.5

    # === Case C: Norm/Bias (Vector) [N] ===
    # Theory: Spectral norm ~ sqrt(2 * ln(N)) (Logarithmic Law)
    elif len(shape) == 1:
        n_dim = shape[0]
        return math.sqrt(2 * math.log(max(n_dim, 2.0)))

    return None

# ==============================================================================
# 2. Helper Function: Fan Dims for LR Calculation (Structure Analysis)
# ==============================================================================
def get_fan_structure(shape):
    """
    Parses shape into (Fan_Out, Fan_In) purely for structure identification.
    """
    if len(shape) == 2:
        return shape[0], shape[1]
    elif len(shape) == 4:
        return shape[0], shape[1] * shape[2] * shape[3]
    elif len(shape) == 1:
        return shape[0], 1.0 # Treat vector as [N, 1]
    return None

# ==============================================================================
# 3. Main Functions
# ==============================================================================

def _get_clip4target(base_shapes, target_shapes, target_noise=None):
    valid_keys = []
    f_vector_list = []
    
    common_keys = [k for k in base_shapes.keys() if k in target_shapes]
    
    for key in common_keys:
        # === 🟢 FIX: Skip Bias ===
        # Bias shares the clipping budget with weights, and weight norm dominates.
        if key.endswith(".bias"):
            continue

        b_shape = base_shapes[key]
        t_shape = target_shapes[key]
        
        # === 1. Use Raw Spectral Metric (sqrt(d) vs sqrt(log d)) ===
        base_metric = get_raw_spectral_metric(b_shape)
        target_metric = get_raw_spectral_metric(t_shape)
        
        if base_metric is None or target_metric is None: 
            continue

        # === 2. Compute Ratio ===
        # Ratio describes how the "sensitivity" (spectral norm) scales
        f_vector_list.append((base_metric / target_metric) ** 2)
        valid_keys.append(key)

    L = len(f_vector_list)
    print(f"Effective Layers for Noise (Blocks Only): {L}") 
    if L == 0: return {}
    
    f_vector = torch.tensor(f_vector_list, dtype=torch.float32)
    sum_term = torch.sum(1.0 / f_vector)

    # clip_coeff = 1 / sqrt(f_i * sum(1/f))
    clip_coeff_vector = 1.0 / torch.sqrt(f_vector * sum_term)
    
    return dict(zip(valid_keys, clip_coeff_vector))


def _get_lr4target_adam(base_shapes, target_shapes, base_noise, target_noise, base_lr):
    """
    Calculates Learning Rate for Adam Optimizer (muAdam).
    Theory:
      - Linear Layers (Matrix): Update ~ Sign(G), Spectral Norm ~ sqrt(d). 
        LR needs to scale down by 1/d (or 1/width).
      - Norm Layers (Vector): Update ~ Sign(G), Spectral Norm ~ O(1). 
        LR should be CONSTANT (no scaling).
    """
    target_lrs = {}
    common_keys = [k for k in base_shapes.keys() if k in target_shapes]

    for key in common_keys:
        # === 🟢 FIX: Skip Bias ===
        # Bias usually implicitly follows the LR of its weight or stays constant.
        # In standard muP, Bias LR is often kept constant or scaled differently for SGD.
        # For simple Adam transfer, skipping it (letting it use default LR) or constant is fine.
        if key.endswith(".bias"):
            continue

        b_shape = base_shapes[key]
        t_shape = target_shapes[key]
        
        # === 1. Check Layer Type ===
        # If it is a Vector (Norm Layer), we force Ratio = 1.0 (Constant Transfer)
        # since fan_in = fan_out, sign(diag(v)) = O(1), thus no need to change
        if len(b_shape) == 1:
            target_lrs[key] = base_lr
            continue

        # === 2. Handle Matrix Layers ===
        base_fans = get_fan_structure(b_shape)
        target_fans = get_fan_structure(t_shape)
        
        if base_fans is None or target_fans is None:
            target_lrs[key] = base_lr
            continue

        b_out, b_in = base_fans
        t_out, t_in = target_fans
        
        # === 3. Compute Metrics for Adam (Matrix) ===
        # For matrices, Adam needs to counteract the sqrt(d) growth of the sign update.
        # Metric ~ 1/width
        
        # Heuristic from muP paper (Table 8 / Appendix):
        # phi_base = scale_base / norm_base
        # This roughly equates to 1/d scaling
        norm_base = b_out**0.5 + b_in**0.5
        scale_base = (b_out / b_in) ** 0.5
        
        norm_target = t_out**0.5 + t_in**0.5
        scale_target = (t_out / t_in) ** 0.5

        metric_base = scale_base / norm_base
        metric_target = scale_target / norm_target
        
        ratio = metric_target / metric_base
        target_lrs[key] = base_lr * ratio
        
    return target_lrs


def _get_lr4target(base_shapes, target_shapes, base_noise, target_noise, base_lr):
    """
    Standard SGD Transfer (Not used for Adam, but updated for consistency).
    Depends on Noise Scale explicitly.
    """
    target_lrs = {}
    common_keys = [k for k in base_shapes.keys() if k in target_shapes]

    for key in common_keys:
        if key.endswith(".bias"):
            continue

        base_fans = get_fan_structure(base_shapes[key])
        target_fans = get_fan_structure(target_shapes[key])
        
        if base_fans is None or target_fans is None:
            target_lrs[key] = base_lr
            continue

        b_out, b_in = base_fans
        t_out, t_in = target_fans
        
        # SGD Logic: LR needs to account for Noise and Width
        # norm term depends on noise scale
        norm_base = base_noise * (b_out**0.5 + b_in**0.5)
        scale_base = (b_out / b_in) ** 0.5
        
        norm_target = target_noise * (t_out**0.5 + t_in**0.5)
        scale_target = (t_out / t_in) ** 0.5

        metric_base = scale_base / norm_base
        metric_target = scale_target / norm_target
        
        ratio = metric_target / metric_base
        target_lrs[key] = base_lr * ratio
        
    return target_lrs

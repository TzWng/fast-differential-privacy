import torch
import math

def get_shapes(model):
    """
    Returns a dictionary of {parameter_name: parameter_shape}.
    Used to compare architectures between Base and Target models.
    """
    if hasattr(model, "get_shapes"):
        return model.get_shapes()
    return {name: param.shape for name, param in model.named_parameters()}


# ==============================================================================
# 1. Helper: Spectral Metric for RAW NOISE (for Clipping & Sigma)
# ==============================================================================
def get_raw_spectral_metric(shape):
    """
    Calculates the Spectral Norm of a standard Gaussian Noise matrix/vector.
    
    Theory Used:
    - Matrix (Linear/Conv): Spectral Norm ~ sqrt(N) + sqrt(M) (Marchenko-Pastur Law)
    - Vector (Norm/Bias):   Spectral Norm ~ sqrt(2 * ln(N)) (Extreme Value Theory)
    """
    # Case A: Linear Layer (Matrix) [Out, In]
    if len(shape) == 2:
        n_out, n_in = shape
        return n_out**0.5 + n_in**0.5
    
    # Case B: Conv2d (4D Tensor) -> Treat as Matrix
    elif len(shape) == 4:
        n_out = shape[0]
        n_in = shape[1] * shape[2] * shape[3]
        return n_out**0.5 + n_in**0.5

    # Case C: Norm Layer / Bias (Vector) [N]
    # Note: This grows much slower than matrix layers.
    elif len(shape) == 1:
        n_dim = shape[0]
        # max(..., 2.0) prevents math domain error for small N
        return math.sqrt(2 * math.log(max(n_dim, 2.0)))

    return None


# ==============================================================================
# 2. Helper: Fan Structure for LR (for Architecture Analysis)
# ==============================================================================
def get_fan_structure(shape):
    """
    Parses shape into (Fan_Out, Fan_In) purely for structure identification.
    Returns: (fan_out, fan_in) or None
    """
    if len(shape) == 2:
        return shape[0], shape[1]
    elif len(shape) == 4:
        return shape[0], shape[1] * shape[2] * shape[3]
    elif len(shape) == 1:
        return shape[0], 1.0 # Treat vector as [N, 1]
    return None


# ==============================================================================
# 3. Core Function: Get Target Noise (Sigma)
# ==============================================================================
def _get_noise4target(base_shapes, target_shapes, base_noise):
    """
    Calculates the optimal noise (sigma) for the target model.
    Based on the equation system where Linear layers scale with sqrt(d) 
    and Norm layers scale with sqrt(ln(d)).
    """
    f_vector_list = []
    
    # Only process layers that exist in both models
    common_keys = [k for k in base_shapes.keys() if k in target_shapes]
    
    for key in common_keys:
        # === Skip Bias ===
        # Bias shares the clipping budget with its weight. 
        # Including it would artificially inflate the layer count (L).
        if key.endswith(".bias"):
            continue

        b_shape = base_shapes[key]
        t_shape = target_shapes[key]

        # 1. Get Metric based on Geometry (Matrix vs Vector)
        base_metric = get_raw_spectral_metric(b_shape)
        target_metric = get_raw_spectral_metric(t_shape)
        
        if base_metric is None or target_metric is None:
            continue
            
        # 2. Compute Sensitivity Ratio^2
        # Ratio = (Base_Metric / Target_Metric)^2
        ratio = (base_metric / target_metric) ** 2
        f_vector_list.append(ratio)
        
    L = len(f_vector_list)
    print(f"[muP] Effective Layers for Noise Calculation (Weights Only): {L}")
    
    if L == 0: return base_noise 

    f_vector = torch.tensor(f_vector_list, dtype=torch.float32)
    
    # 3. Solve for Sigma'
    # The formula ensures total privacy budget consumption is invariant.
    sum_term = torch.sum(1.0 / f_vector)
    target_noise = base_noise / (sum_term / L) ** 0.5

    return target_noise.item()


# ==============================================================================
# 4. Core Function: Get Target Clipping Coefficients
# ==============================================================================
def _get_clip4target(base_shapes, target_shapes, target_noise=None):
    """
    Calculates layer-wise clipping coefficients (D_i).
    Linear layers will get tighter clipping as width increases.
    Norm layers will maintain relatively larger clipping thresholds.
    """
    valid_keys = []
    f_vector_list = []
    
    common_keys = [k for k in base_shapes.keys() if k in target_shapes]
    
    for key in common_keys:
        # === Skip Bias ===
        # Bias will implicitly use the D_i of the corresponding weight or global default
        if key.endswith(".bias"):
            continue

        b_shape = base_shapes[key]
        t_shape = target_shapes[key]
        
        base_metric = get_raw_spectral_metric(b_shape)
        target_metric = get_raw_spectral_metric(t_shape)
        
        if base_metric is None or target_metric is None: 
            continue

        # Ratio of sensitivities
        f_vector_list.append((base_metric / target_metric) ** 2)
        valid_keys.append(key)

    L = len(f_vector_list)
    if L == 0: return {}
    
    f_vector = torch.tensor(f_vector_list, dtype=torch.float32)
    sum_term = torch.sum(1.0 / f_vector)

    # Solve for D_i
    clip_coeff_vector = 1.0 / torch.sqrt(f_vector * sum_term)
    
    return dict(zip(valid_keys, clip_coeff_vector))


# ==============================================================================
# 5. Core Function: Get Target LRs for ADAM (muAdam)
# ==============================================================================
def _get_lr4target_adam(base_shapes, target_shapes, base_noise, target_noise, base_lr):
    """
    Calculates Learning Rates for Adam Optimizer.
    
    CRITICAL THEORY:
    - Linear Layers (Matrix): Sign(G) has spectral norm ~ sqrt(d). 
      -> LR must scale down (~ 1/d).
    - Norm Layers (Vector): Sign(G) has spectral norm ~ O(1).
      -> LR must stay CONSTANT.
    """
    target_lrs = {}
    common_keys = [k for k in base_shapes.keys() if k in target_shapes]

    for key in common_keys:
        # === Skip Bias ===
        # Bias follows Weight LR or uses Base LR.
        if key.endswith(".bias"):
            continue

        b_shape = base_shapes[key]
        t_shape = target_shapes[key]
        
        # === Case 1: Norm/Vector Layers ===
        # Logic: Sign update spectral norm is constant (1).
        # since fan_in = fan_out, sign(diag(v)) = O(1), no need to change
        if len(b_shape) == 1:
            target_lrs[key] = base_lr
            continue

        # === Case 2: Linear/Matrix Layers ===
        # Logic: Sign update spectral norm grows with sqrt(d).
        # Action: Scale LR.
        base_fans = get_fan_structure(b_shape)
        target_fans = get_fan_structure(t_shape)
        
        if base_fans is None or target_fans is None:
            target_lrs[key] = base_lr
            continue

        b_out, b_in = base_fans
        t_out, t_in = target_fans
        
        # Heuristic for Adam Scaling (approx 1/width)
        norm_base = b_out**0.5 + b_in**0.5
        scale_base = (b_out / b_in) ** 0.5
        
        norm_target = t_out**0.5 + t_in**0.5
        scale_target = (t_out / t_in) ** 0.5

        metric_base = scale_base / norm_base
        metric_target = scale_target / norm_target
        
        ratio = metric_target / metric_base
        target_lrs[key] = base_lr * ratio
        
    return target_lrs


# ==============================================================================
# 6. Core Function: Get Target LRs for SGD (muSGD)
# ==============================================================================
def _get_lr4target(base_shapes, target_shapes, base_noise, target_noise, base_lr):
    """
    Standard SGD Transfer logic.
    Unlike Adam, SGD scaling explicitly depends on Noise Scale.
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
        norm_base = base_noise * (b_out**0.5 + b_in**0.5)
        scale_base = (b_out / b_in) ** 0.5
        
        norm_target = target_noise * (t_out**0.5 + t_in**0.5)
        scale_target = (t_out / t_in) ** 0.5

        metric_base = scale_base / norm_base
        metric_target = scale_target / norm_target
        
        ratio = metric_target / metric_base
        target_lrs[key] = base_lr * ratio
        
    return target_lrs

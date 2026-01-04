import torch

def get_shapes(model):
    # If you want to implement a custom shapes function, you can use this name
    if hasattr(model, "get_shapes"):
        return model.get_shapes()
    return {name: param.shape for name, param in model.named_parameters()}


def get_fan_dims(shape):
    """
    Unified parser for Tensor shapes into (Fan_Out, Fan_In).
    
    Args:
        shape: torch.Size or list
        
    Returns:
        (fan_out, fan_in): float or int
        None: If shape cannot be handled (e.g., 3D PosEmbed)
    """
    # Case A: Standard Linear Layer [Out, In]
    if len(shape) == 2:
        return shape[0], shape[1]
    
    # Case B: 4D Tensor (Conv2d / PatchEmbed) [Out, In, H, W]
    # Logic: Treated as [Out, In * H * W]
    elif len(shape) == 4:
        return shape[0], shape[1] * shape[2] * shape[3]
    
    # Case C: 1D Tensor (Bias, LayerNorm) [N]
    # Logic: Treated as [N, 1] matrix
    elif len(shape) == 1:
        return shape[0], 1.0
        
    # Case D: Others (e.g., 3D PosEmbed), ignore for now
    return None
    

def _get_noise4target(base_shapes, target_shapes, base_noise):
    f_vector_list = []
    
    common_keys = [k for k in base_shapes.keys() if k in target_shapes]
    
    for key in common_keys:
        b_shape = base_shapes[key]
        t_shape = target_shapes[key]

        # === 1. Use unified function to parse dimensions ===
        # This now handles 1D layers (Bias/Norm) correctly as well
        base_fans = get_fan_dims(b_shape)
        target_fans = get_fan_dims(t_shape)
        
        if base_fans is None or target_fans is None:
            continue
            
        b_out, b_in = base_fans
        t_out, t_in = target_fans
        
        # === 2. Compute Metric ===
        # Metric = sqrt(fan_out) + sqrt(fan_in)
        base_metric = b_out ** 0.5 + b_in ** 0.5
        target_metric = t_out ** 0.5 + t_in ** 0.5
        
        ratio = (base_metric / target_metric) ** 2
        f_vector_list.append(ratio)
        
    L = len(f_vector_list)
    print(f"Effective Layers for Noise (All Layers): {L}") # Should be 64 for n_layer=10
    
    if L == 0: return base_noise # Fallback

    f_vector = torch.tensor(f_vector_list, dtype=torch.float32)
    sum_term = torch.sum(1.0 / f_vector)
    target_noise = base_noise / (sum_term / L) ** 0.5

    return target_noise.item()


def _get_clip4target(base_shapes, target_shapes, target_noise=None):
    valid_keys = []
    f_vector_list = []
    
    common_keys = [k for k in base_shapes.keys() if k in target_shapes]
    
    for key in common_keys:
        b_shape = base_shapes[key]
        t_shape = target_shapes[key]
        
        # === 1. Use unified function ===
        base_fans = get_fan_dims(b_shape)
        target_fans = get_fan_dims(t_shape)
        
        if base_fans is None or target_fans is None: continue

        b_out, b_in = base_fans
        t_out, t_in = target_fans
        
        # === 2. Compute Metric ===
        base_metric = b_out ** 0.5 + b_in ** 0.5
        target_metric = t_out ** 0.5 + t_in ** 0.5
        
        f_vector_list.append((base_metric / target_metric) ** 2)
        valid_keys.append(key)

    L = len(f_vector_list)
    if L == 0: return {}
    
    f_vector = torch.tensor(f_vector_list, dtype=torch.float32)
    sum_term = torch.sum(1.0 / f_vector)

    # clip_coeff = 1 / sqrt(f_i * sum(1/f))
    clip_coeff_vector = 1.0 / torch.sqrt(f_vector * sum_term)
    
    return dict(zip(valid_keys, clip_coeff_vector))


def _get_lr4target(base_shapes, target_shapes, base_noise, target_noise, base_lr):
    target_lrs = {}
    common_keys = [k for k in base_shapes.keys() if k in target_shapes]

    for key in common_keys:
        # === 1. Use unified function ===
        base_fans = get_fan_dims(base_shapes[key])
        target_fans = get_fan_dims(target_shapes[key])
        
        # If None (e.g., PosEmbed), keep base_lr
        if base_fans is None or target_fans is None:
            target_lrs[key] = base_lr
            continue

        b_out, b_in = base_fans
        t_out, t_in = target_fans
        
        # === 2. Compute Metrics (SGD/Shampoo logic) ===
        # norm term depends on noise scale
        norm_base = base_noise * (b_out**0.5 + b_in**0.5)
        scale_base = (b_out / b_in) ** 0.5
        
        norm_target = target_noise * (t_out**0.5 + t_in**0.5)
        scale_target = (t_out / t_in) ** 0.5

        # === 3. Scaling Ratio ===
        # LR ~ scale / norm
        metric_base = scale_base / norm_base
        metric_target = scale_target / norm_target
        
        ratio = metric_target / metric_base
        target_lrs[key] = base_lr * ratio
        
    return target_lrs


def _get_lr4target_adam(base_shapes, target_shapes, base_noise, target_noise, base_lr):
    target_lrs = {}
    common_keys = [k for k in base_shapes.keys() if k in target_shapes]

    for key in common_keys:
        # === 1. Use unified function ===
        base_fans = get_fan_dims(base_shapes[key])
        target_fans = get_fan_dims(target_shapes[key])
        
        if base_fans is None or target_fans is None:
            target_lrs[key] = base_lr
            continue

        b_out, b_in = base_fans
        t_out, t_in = target_fans
        
        # === 2. Compute Metrics (Adam logic) ===
        # Adam is scale invariant to gradient norm, so noise doesn't appear here
        # But we still scale based on layer dimensions
        norm_base = b_out**0.5 + b_in**0.5
        scale_base = (b_out / b_in) ** 0.5
        
        norm_target = t_out**0.5 + t_in**0.5
        scale_target = (t_out / t_in) ** 0.5

        # === 3. Scaling Ratio ===
        metric_base = scale_base / norm_base
        metric_target = scale_target / norm_target
        
        ratio = metric_target / metric_base
        target_lrs[key] = base_lr * ratio
        
    return target_lrs

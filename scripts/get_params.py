import torch

def get_shapes(model):
    # If you want to implement a custom shapes function, you can use this name
    if hasattr(model, "get_shapes"):
        return model.get_shapes()
    return {name: param.shape for name, param in model.named_parameters()}


def get_fan_dims(shape):
    """
    统一解析 Tensor 形状为 (Fan_Out, Fan_In)。
    
    Args:
        shape: torch.Size or list
        
    Returns:
        (fan_out, fan_in): float or int
        None: 如果形状无法处理 (如 3D PosEmbed)
    """
    # Case A: 标准 Linear 层 [Out, In]
    if len(shape) == 2:
        return shape[0], shape[1]
    
    # Case B: 4D Tensor (Conv2d / PatchEmbed) [Out, In, H, W]
    # 逻辑: 视为 [Out, In * H * W]
    elif len(shape) == 4:
        return shape[0], shape[1] * shape[2] * shape[3]
    
    # Case C: 1D Tensor (Bias, LayerNorm) [N]
    # 逻辑: 视为 [N, 1] 的矩阵
    elif len(shape) == 1:
        return shape[0], 1.0
        
    # Case D: 其他 (如 3D PosEmbed), 暂不处理
    return None
    

def _get_noise4target(base_shapes, target_shapes, base_noise):
    f_vector_list = []
    
    common_keys = [k for k in base_shapes.keys() if k in target_shapes]
    
    for key in common_keys:
        b_shape = base_shapes[key]
        t_shape = target_shapes[key]

        # === 1. 过滤非 Matrix 层 (Bias/Norm) ===
        # Noise Scaling 通常只针对权重矩阵，加入 Bias 会导致分母虚高
        if len(b_shape) < 2: 
            continue

        # === 2. 调用公共函数解析维度 ===
        base_fans = get_fan_dims(b_shape)
        target_fans = get_fan_dims(t_shape)
        
        if base_fans is None or target_fans is None:
            continue
            
        b_out, b_in = base_fans
        t_out, t_in = target_fans
        
        # === 3. 计算 Metric ===
        base_metric = b_out ** 0.5 + b_in ** 0.5
        target_metric = t_out ** 0.5 + t_in ** 0.5
        
        ratio = (base_metric / target_metric) ** 2
        f_vector_list.append(ratio)
        
    L = len(f_vector_list)
    print(f"Effective Layers for Noise (Matrix Only): {L}") # 应该是 50
    
    if L == 0: return base_noise # Fallback

    f_vector = torch.tensor(f_vector_list, dtype=torch.float32)
    sum_term = torch.sum(1.0 / f_vector)
    target_noise = base_noise / (sum_term / L) ** 0.5

    return target_noise

def _get_clip4target(base_shapes, target_shapes, target_noise=None):
    valid_keys = []
    f_vector_list = []
    
    common_keys = [k for k in base_shapes.keys() if k in target_shapes]
    
    for key in common_keys:
        b_shape = base_shapes[key]
        t_shape = target_shapes[key]
        
        # === 1. 过滤非 Matrix 层 ===
        if len(b_shape) < 2:
            continue
            
        # === 2. 调用公共函数 ===
        base_fans = get_fan_dims(b_shape)
        target_fans = get_fan_dims(t_shape)
        
        if base_fans is None or target_fans is None: continue

        b_out, b_in = base_fans
        t_out, t_in = target_fans
        
        # === 3. 计算 Metric ===
        base_metric = b_out ** 0.5 + b_in ** 0.5
        target_metric = t_out ** 0.5 + t_in ** 0.5
        
        f_vector_list.append((base_metric / target_metric) ** 2)
        valid_keys.append(key)

    L = len(f_vector_list)
    if L == 0: return {}
    
    f_vector = torch.tensor(f_vector_list, dtype=torch.float32)
    sum_term = torch.sum(1.0 / f_vector)

    clip_coeff_vector = 1.0 / torch.sqrt(f_vector * sum_term)
    
    return dict(zip(valid_keys, clip_coeff_vector))

def _get_lr4target(base_shapes, target_shapes, base_noise, target_noise, base_lr):
    target_lrs = {}
    common_keys = [k for k in base_shapes.keys() if k in target_shapes]

    for key in common_keys:
        # === 1. 调用公共函数 (包含 Bias/Norm 的处理) ===
        base_fans = get_fan_dims(base_shapes[key])
        target_fans = get_fan_dims(target_shapes[key])
        
        # 如果是 None (比如 PosEmbed), 保持 base_lr
        if base_fans is None or target_fans is None:
            target_lrs[key] = base_lr
            continue

        b_out, b_in = base_fans
        t_out, t_in = target_fans
        
        # === 2. 计算 Metrics ===
        # 这里 Bias 会被当作 [N, 1] 矩阵参与计算
        norm_base = base_noise * (b_out**0.5 + b_in**0.5)
        scale_base = (b_out / b_in) ** 0.5
        
        norm_target = target_noise * (t_out**0.5 + t_in**0.5)
        scale_target = (t_out / t_in) ** 0.5

        # === 3. Scaling Ratio ===
        metric_base = scale_base / norm_base
        metric_target = scale_target / norm_target
        
        ratio = metric_target / metric_base
        target_lrs[key] = base_lr * ratio
        
    return target_lrs

# def _get_lr4target(target_shapes, target_noise, base_lr):
#     """
#     Scale the learning rate based solely on the Target model's shape and noise.
#     No comparison with the Base model is needed.
    
#     Rule: LR = base_lr * AspectRatio / Metric
#     """
#     target_lrs = {}

#     # Directly iterate through target_shapes
#     for key, shape in target_shapes.items():
        
#         # 1. Filter out non-weight parameters (e.g., Bias [512], LayerNorm [512])
#         # These parameters usually don't have out/in dimensions or don't need such aggressive scaling
#         if len(shape) < 2:
#             target_lrs[key] = base_lr
#             continue

#         # 2. Extract dimensions [out_features, in_features]
#         t_out, t_in = shape[0], shape[1]
        
#         # 3. Calculate the denominator Metric
#         # Metric = (sqrt(out) + sqrt(in)) * noise
#         layer_metric = (t_out**0.5 + t_in**0.5) * target_noise
        
#         # Calculate new LR combined with aspect ratio scaling
#         new_lr = base_lr * (t_out / t_in) ** 0.5 / (layer_metric + 1e-8)
        
#         # 4. Assign LR
#         # Added 1e-8 to prevent division by zero error if noise is 0
#         target_lrs[key] = new_lr

#     return target_lrs

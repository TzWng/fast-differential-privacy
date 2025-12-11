import torch

def get_shapes(model):
    # If you want to implement a custom shapes function, you can use this name
    if hasattr(model, "get_shapes"):
        return model.get_shapes()
    return {name: param.shape for name, param in model.named_parameters()}
    
def _get_noise4base(base_shapes, target_shapes, target_noise):
    # 1. 找出两个模型共有的层（Key）
    common_keys = [k for k in base_shapes.keys() if k in target_shapes]
    L = len(common_keys)
    
    # 如果没有共有的层，防止除零错误
    if L == 0:
        raise ValueError("两个模型没有共有的层名称 (Keys)！")

    f_vector = torch.zeros(L, dtype=torch.float32)
    
    # 2. 遍历共有的层
    for i, key in enumerate(common_keys):
        b_shape = base_shapes[key]
        t_shape = target_shapes[key]
        
        # 计算 (sqrt(in) + sqrt(out)) 的比值
        # 注意：这里假设 shape 至少有两维 [out, in]
        if len(b_shape) < 2: 
             # 遇到 bias 或一维参数时跳过或处理，这里简单给1.0避免报错，或者你可以选择 continue
            f_vector[i] = 1.0 
            continue

        base_dim_metric = b_shape[0] ** 0.5 + b_shape[1] ** 0.5
        target_dim_metric = t_shape[0] ** 0.5 + t_shape[1] ** 0.5
        f_vector[i] = (base_dim_metric / target_dim_metric) ** 2
        
    sum_term = torch.sum(1.0 / f_vector)
    base_noise = target_noise * (sum_term / L) ** 0.5

    return base_noise


import torch

def _get_noise4target(base_shapes, target_shapes, base_noise):
    # 1. 找出两个模型共有的层（Key）
    common_keys = [k for k in base_shapes.keys() if k in target_shapes]
    L = len(common_keys)
    
    # 如果没有共有的层，防止除零错误
    if L == 0:
        raise ValueError("两个模型没有共有的层名称 (Keys)！")

    f_vector = torch.zeros(L, dtype=torch.float32)
    
    # 2. 遍历共有的层
    for i, key in enumerate(common_keys):
        b_shape = base_shapes[key]
        t_shape = target_shapes[key]
        
        # 计算 (sqrt(in) + sqrt(out)) 的比值
        # 注意：这里假设 shape 至少有两维 [out, in]
        if len(b_shape) < 2: 
             # 遇到 bias 或一维参数时跳过或处理，这里简单给1.0避免报错，或者你可以选择 continue
            f_vector[i] = 1.0 
            continue

        base_dim_metric = b_shape[0] ** 0.5 + b_shape[1] ** 0.5
        target_dim_metric = t_shape[0] ** 0.5 + t_shape[1] ** 0.5
        f_vector[i] = (base_dim_metric / target_dim_metric) ** 2
        
    sum_term = torch.sum(1.0 / f_vector)
    target_noise = base_noise / (sum_term / L) ** 0.5

    return target_noise

def _get_clip4target(base_shapes, target_shapes, target_noise=None):
    common_keys = [k for k in base_shapes.keys() if k in target_shapes]
    L = len(common_keys)
    
    if L == 0: return {} # 返回空字典

    f_vector = torch.zeros(L, dtype=torch.float32)
    
    for i, key in enumerate(common_keys):
        b_shape = base_shapes[key]
        t_shape = target_shapes[key]
        
        if len(b_shape) < 2:
            f_vector[i] = 1.0
            continue
            
        base_dim_metric = b_shape[0] ** 0.5 + b_shape[1] ** 0.5
        target_dim_metric = t_shape[0] ** 0.5 + t_shape[1] ** 0.5
        f_vector[i] = (base_dim_metric / target_dim_metric) ** 2
        
    sum_term = torch.sum(1.0 / f_vector)
    
    # 计算 Vector
    D_prime_vector = 1.0 / (f_vector * sum_term) ** 0.5
    
    # 3. 将结果打包回字典，方便查看每一层对应的 clip
    return dict(zip(common_keys, D_prime_vector))


import torch

def _get_lr4target(base_shapes, target_shapes, base_noise, target_noise, base_lr):
    """
    根据形状变化和差分隐私噪声系数，为 Target 模型计算每一层的特定 Learning Rate。
    """
    target_lrs = {}

    for key, base_shape in base_shapes.items():
        # 1. 基础检查：跳过不匹配的层
        if key not in target_shapes:
            continue
        
        target_shape = target_shapes[key]
        
        # 确保维度一致且至少是 2D (排除 bias 或 LayerNorm 参数，通常它们不需要这种复杂的缩放)
        if len(base_shape) != len(target_shape) or len(base_shape) < 2:
            # 对于 bias 或 1D 参数，通常保持原 base_lr 或按需处理
            # 这里暂时设为 base_lr，你也可以选择跳过
            target_lrs[key] = base_lr
            continue 

        # 2. 提取维度 (无需循环维度，直接取 Out/In)
        # 假设 shape 格式为 [out_features, in_features]
        # d_out, d_in
        b_out, b_in = base_shape[0], base_shape[1]
        t_out, t_in = target_shape[0], target_shape[1]

        # 3. 计算 Base 模型的指标
        # Norm: 与噪声和维度和有关 (类似 DP 梯度裁剪范数的影响)
        norm_base = base_noise * (b_out**0.5 + b_in**0.5)
        # Scale: 维度的纵横比 (Aspect Ratio / Fan-out vs Fan-in)
        scale_base = (b_out / b_in) ** 0.5
        
        # 4. 计算 Target 模型的指标
        norm_target = target_noise * (t_out**0.5 + t_in**0.5)
        scale_target = (t_out / t_in) ** 0.5

        # 5. 计算缩放比例
        # 逻辑：维持 (Scale / Norm) 的比例一致性
        # Ratio = Target_Metric / Base_Metric
        metric_target = scale_target / norm_target
        metric_base = scale_base / norm_base
        
        ratio = metric_target / metric_base
        
        # 6. 得到最终 LR
        target_lrs[key] = base_lr * ratio
        
    return target_lrs


    

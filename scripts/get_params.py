import torch

def get_shapes(model):
    # If you want to implement a custom shapes function, you can use this name
    if hasattr(model, "get_shapes"):
        return model.get_shapes()
    return {name: param.shape for name, param in model.named_parameters()}
    
def _get_noise4base(base_shapes, target_shapes, target_noise):
    # 1. Identify shared layers (Keys) between the two models
    common_keys = [k for k in base_shapes.keys() if k in target_shapes]
    L = len(common_keys)
    
    # Prevent division by zero error if there are no shared layers
    if L == 0:
        raise ValueError("The two models share no common layer names (Keys)!")

    f_vector = torch.zeros(L, dtype=torch.float32)
    
    # 2. Iterate through shared layers
    for i, key in enumerate(common_keys):
        b_shape = base_shapes[key]
        t_shape = target_shapes[key]
        
        # Calculate the ratio based on dimensions
        # Note: Assumes shape has at least two dimensions [out, in]
        if len(b_shape) < 2: 
             # Skip or handle 1D parameters (like bias). 
             # Here we assign 1.0 to avoid errors, or you could use continue.
            f_vector[i] = 1.0 
            continue

        base_dim_metric = b_shape[0] ** 0.5 + b_shape[1] ** 0.5
        target_dim_metric = t_shape[0] ** 0.5 + t_shape[1] ** 0.5
        f_vector[i] = (base_dim_metric / target_dim_metric) ** 2
        
    sum_term = torch.sum(1.0 / f_vector)
    base_noise = target_noise * (sum_term / L) ** 0.5

    return base_noise


def _get_noise4target(base_shapes, target_shapes, base_noise):
    # 1. Identify shared layers (Keys) between the two models
    common_keys = [k for k in base_shapes.keys() if k in target_shapes]
    L = len(common_keys)
    
    # Prevent division by zero error if there are no shared layers
    if L == 0:
        raise ValueError("The two models share no common layer names (Keys)!")

    f_vector = torch.zeros(L, dtype=torch.float32)
    
    # 2. Iterate through shared layers
    for i, key in enumerate(common_keys):
        b_shape = base_shapes[key]
        t_shape = target_shapes[key]
        
        # Calculate the ratio based on dimensions
        # Note: Assumes shape has at least two dimensions [out, in]
        if len(b_shape) < 2: 
             # Skip or handle 1D parameters (like bias). 
             # Here we assign 1.0 to avoid errors, or you could use continue.
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
    
    if L == 0: return {} # Return empty dictionary

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
    
    # Calculate Vector
    D_prime_vector = 1.0 / (f_vector * sum_term) ** 0.5
    
    # 3. Package results back into a dictionary for easy inspection of layer-wise clips
    return dict(zip(common_keys, D_prime_vector))


# def _get_lr4target(base_shapes, target_shapes, base_noise, target_noise, base_lr):
#     """
#     Calculate specific Learning Rate for each layer in the Target model 
#     based on shape changes and Differential Privacy noise coefficients.
#     """
#     target_lrs = {}

#     for key, base_shape in base_shapes.items():
#         # 1. Basic check: Skip layers missing in target
#         if key not in target_shapes:
#             continue
        
#         target_shape = target_shapes[key]
        
#         # Ensure dimensions match and are at least 2D 
#         # (Exclude bias or LayerNorm parameters, they usually don't need this complex scaling)
#         if len(base_shape) != len(target_shape) or len(base_shape) < 2:
#             # For bias or 1D parameters, usually keep original base_lr or handle as needed
#             # Here we temporarily set to base_lr, or you can choose to skip
#             target_lrs[key] = base_lr
#             continue 

#         # 2. Extract dimensions (Directly take Out/In without looping)
#         # Assume shape format is [out_features, in_features]
#         b_out, b_in = base_shape[0], base_shape[1]
#         t_out, t_in = target_shape[0], target_shape[1]

#         # 3. Calculate Base Model Metrics
#         # Norm: Related to noise and sum of dimensions (similar to DP gradient clipping norm impact)
#         norm_base = base_noise * (b_out**0.5 + b_in**0.5)
#         # Scale: Aspect Ratio of dimensions (Fan-out vs Fan-in)
#         scale_base = (b_out / b_in) ** 0.5
        
#         # 4. Calculate Target Model Metrics
#         norm_target = target_noise * (t_out**0.5 + t_in**0.5)
#         scale_target = (t_out / t_in) ** 0.5

#         # 5. Calculate Scaling Ratio
#         # Logic: Maintain the consistency of (Scale / Norm)
#         # Ratio = Target_Metric / Base_Metric
#         metric_target = scale_target / norm_target
#         metric_base = scale_base / norm_base
        
#         ratio = metric_target / metric_base
    
#         # 6. Get Final LR
#         target_lrs[key] = base_lr * ratio
        
#     return target_lrs


def _get_lr4target(target_shapes, target_noise, base_lr):
    """
    Scale the learning rate based solely on the Target model's shape and noise.
    No comparison with the Base model is needed.
    
    Rule: LR = base_lr * AspectRatio / Metric
    """
    target_lrs = {}

    # Directly iterate through target_shapes
    for key, shape in target_shapes.items():
        
        # 1. Filter out non-weight parameters (e.g., Bias [512], LayerNorm [512])
        # These parameters usually don't have out/in dimensions or don't need such aggressive scaling
        if len(shape) < 2:
            target_lrs[key] = base_lr
            continue

        # 2. Extract dimensions [out_features, in_features]
        t_out, t_in = shape[0], shape[1]
        
        # 3. Calculate the denominator Metric
        # Metric = (sqrt(out) + sqrt(in)) * noise
        layer_metric = (t_out**0.5 + t_in**0.5) * target_noise
        
        # Calculate new LR combined with aspect ratio scaling
        new_lr = base_lr * (t_out / t_in) ** 0.5 / (layer_metric + 1e-8)
        
        # 4. Assign LR
        # Added 1e-8 to prevent division by zero error if noise is 0
        target_lrs[key] = new_lr

    return target_lrs

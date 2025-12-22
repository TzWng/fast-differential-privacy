import torch
import torch.nn as nn
import timm

class MyVit:
    """
    专门用于构建无 Input/Output Bias 的 ViT 模型工厂类。
    用于验证 Scaling Law 和 SNR 理论，消除常数项干扰。
    """
    def __init__(self, args, is_base=False):
        """
        :param args: 全局参数
        :param is_base: 如果为 True，强制 scale=1.0 (Base Model)；否则使用 args.scale (Target Model)
        """
        self.args = args
        self.is_base = is_base
        
        # 解析类别数
        self.num_classes = int(args.cifar_data[5:]) if hasattr(args, 'cifar_data') else 10

        # === 核心逻辑 ===
        if self.is_base:
            self.current_scale = 1.0
            mode_str = "Base Model (Scale 1.0)"
        else:
            # Target Model 跟随 args.scale
            self.current_scale = args.scale
            mode_str = f"Target Model (Scale {self.current_scale})"
        
        print(f"==> [Builder] Configuring {mode_str}...")

        # 计算具体的维度
        self.embed_dim = int(192 * self.current_scale)
        self.num_heads = int(6 * self.current_scale)
        self.mlp_ratio = 4.0

    def create_model(self):
        """
        构建并返回修改后的模型
        :param init_fn: 初始化函数 (e.g., kaiming_init_weights)
        """
        print(f'==> [Builder] Creating {self.args.model} | Scale: {self.args.scale} | Bias-Free Mode')
        print(f'    Dimensions -> Embed: {self.embed_dim}, Heads: {self.num_heads}, Classes: {self.num_classes}')

        # 1. 创建基础 timm 模型
        model = timm.create_model(
            self.args.model,
            pretrained=False,
            num_classes=self.num_classes,
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            mlp_ratio=self.mlp_ratio
        )

        # 2. 执行去 Bias 手术
        self._remove_input_bias(model)
        self._remove_head_bias(model)

        return model

    def _remove_input_bias(self, model):
        """移除 Patch Embedding 层的 Bias"""
        if hasattr(model, 'patch_embed') and hasattr(model.patch_embed, 'proj'):
            old_proj = model.patch_embed.proj
            if isinstance(old_proj, nn.Conv2d):
                # 创建无 Bias 的 Conv2d
                new_proj = nn.Conv2d(
                    in_channels=old_proj.in_channels,
                    out_channels=old_proj.out_channels,
                    kernel_size=old_proj.kernel_size,
                    stride=old_proj.stride,
                    padding=old_proj.padding,
                    bias=False  # <--- Key Change
                )
                # 暂时保留旧权重 (会被后续 init_fn 覆盖，但为了安全先 copy)
                with torch.no_grad():
                    new_proj.weight.copy_(old_proj.weight)
                
                model.patch_embed.proj = new_proj
                print("    [Info] Removed Bias from patch_embed.proj")

    def _remove_head_bias(self, model):
        """移除 Classification Head 的 Bias"""
        if hasattr(model, 'head') and isinstance(model.head, nn.Linear):
            old_head = model.head
            # 创建无 Bias 的 Linear
            new_head = nn.Linear(
                in_features=old_head.in_features,
                out_features=old_head.out_features,
                bias=False  # <--- Key Change
            )
            # 暂时保留旧权重
            with torch.no_grad():
                new_head.weight.copy_(old_head.weight)
                
            model.head = new_head
            print("    [Info] Removed Bias from head")

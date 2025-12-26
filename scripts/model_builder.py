import torch
import torch.nn as nn
import timm

class MyVit:
    def __init__(self, args, is_base=False):
        """
        :param args: 全局参数
        :param is_base: 如果为 True，强制 scale=1.0 (Base Model)；否则使用 args.scale (Target Model)
        """
        self.args = args
        self.is_base = is_base
        
        # 1. 解析类别数 (只做这一件事)
        self.num_classes = self._get_num_classes(args)
        print(f"==> [Pretrain Factory] Dataset: {args.dataset} | Classes: {self.num_classes}")

        # 2. === 核心逻辑：配置模型维度 (搬到这里来！) ===
        if self.is_base:
            self.current_scale = 5.0
            mode_str = "Base Model (Scale 1.0)"
        else:
            # Target Model 跟随 args.scale
            self.current_scale = args.scale
            mode_str = f"Target Model (Scale {self.current_scale})"
        
        print(f"==> [Builder] Configuring {mode_str}...")

        # 计算具体的维度
        # 注意：这里假设 Base 是 Tiny (192, 6)，如果你是 Small Base，这里基准要改
        self.embed_dim = int(192 * self.current_scale)
        self.num_heads = int(6 * self.current_scale)
        self.mlp_ratio = 4.0

    def _get_num_classes(self, args):
        """这个函数只负责返回类别数，不要在里面设置 self 属性"""
        # 优先检查 dataset_name 是否存在
        if not hasattr(args, 'dataset'):
            print("Warning: args.dataset not found. Defaulting to 100 classes.")
            return 100
            
        name = args.dataset
        
        if name in ['SVHN', 'CIFAR10']:
            return 10
        elif name in ['CIFAR100', 'FGVCAircraft']:
            return 100
        elif name in ['Food101']:
            return 101
        elif name in ['GTSRB']:
            return 43
        elif name in ['CelebA']:
            return 40
        elif name in ['Places365']:
            return 365
        elif name in ['ImageNet']:
            return 1000
        elif name in ['INaturalist']:
            return 10000
        
        return 100 # 保底

    def create_model(self):
        print(f'==> [Builder] Creating {self.args.model} | Scale: {self.current_scale} | Bias-Free Mode')
        # 现在这里可以安全访问 self.embed_dim 了
        print(f'    Dimensions -> Embed: {self.embed_dim}, Heads: {self.num_heads}, Classes: {self.num_classes}')

        model = timm.create_model(
            self.args.model,
            pretrained=False,
            num_classes=self.num_classes,
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            mlp_ratio=self.mlp_ratio
        )

        self._remove_input_bias(model)
        self._remove_head_bias(model)

        return model

    def _remove_input_bias(self, model):
        if hasattr(model, 'patch_embed') and hasattr(model.patch_embed, 'proj'):
            old_proj = model.patch_embed.proj
            if isinstance(old_proj, nn.Conv2d):
                new_proj = nn.Conv2d(
                    old_proj.in_channels, old_proj.out_channels,
                    old_proj.kernel_size, old_proj.stride, old_proj.padding,
                    bias=False 
                )
                with torch.no_grad():
                    new_proj.weight.copy_(old_proj.weight)
                model.patch_embed.proj = new_proj
                print("    [Info] Removed Bias from patch_embed.proj")

    def _remove_head_bias(self, model):
        if hasattr(model, 'head') and isinstance(model.head, nn.Linear):
            old_head = model.head
            new_head = nn.Linear(
                old_head.in_features, old_head.out_features, bias=False
            )
            with torch.no_grad():
                new_head.weight.copy_(old_head.weight)
            model.head = new_head
            print("    [Info] Removed Bias from head")



class MyPreVit:
    def __init__(self, args, model_name=None):
        """
        :param args: 必须包含 args.dataset_name (str)
        :param model_name: 强制指定的模型名 (如 'vit_tiny_patch16_224')
        """
        self.args = args
        self.model_name = model_name if model_name else args.model
        
        # === 使用你提供的精确逻辑获取类别数 ===
        self.num_classes = self._get_num_classes(args)
        
        print(f"==> [Pretrain Factory] Dataset: {args.dataset} | Classes: {self.num_classes}")

    def _get_num_classes(self, args):
        # 优先检查 dataset_name 是否存在
        if not hasattr(args, 'dataset'):
            print("Warning: args.dataset not found. Defaulting to 100 classes.")
            return 10
            
        name = args.dataset
        
        # === 你的映射逻辑 ===
        if name in ['SVHN', 'CIFAR10']:
            return 10
        elif name in ['CIFAR100', 'FGVCAircraft']:
            return 100
        elif name in ['Food101']:
            return 101
        elif name in ['GTSRB']:
            return 43
        elif name in ['CelebA']:
            return 40
        elif name in ['Places365']:
            return 365
        elif name in ['ImageNet']:
            return 1000
        elif name in ['INaturalist']:
            return 10000


    def create_model(self):
        print(f"==> [Pretrain Builder] Loading {self.model_name} (ImageNet Weights)...")
        
        # 1. 加载官方预训练模型
        model = timm.create_model(
            self.model_name, 
            pretrained=True
        )
        
        # 2. 手术 I: Input Layer (保留 Weight, 删除 Bias)
        self._remove_input_bias_and_copy_weights(model)

        # 3. 手术 II: Head Layer (重置为 num_classes, 删除 Bias)
        self._rebuild_head_bias_free(model)

        return model

    def _remove_input_bias_and_copy_weights(self, model):
        if not hasattr(model, 'patch_embed') or not hasattr(model.patch_embed, 'proj'):
            return

        old_proj = model.patch_embed.proj
        if isinstance(old_proj, nn.Conv2d):
            new_proj = nn.Conv2d(
                old_proj.in_channels, old_proj.out_channels,
                old_proj.kernel_size, old_proj.stride, old_proj.padding,
                bias=False  # 去除 Bias
            )
            with torch.no_grad():
                new_proj.weight.copy_(old_proj.weight) # 复制权重
            model.patch_embed.proj = new_proj
            print(f"    [Surgery] PatchEmbed bias removed. Weights copied.")

    def _rebuild_head_bias_free(self, model):
        old_head = model.head
        if isinstance(old_head, nn.Linear):
            in_features = old_head.in_features
            
            # 创建新 Head: 目标类别数 self.num_classes, 无 Bias
            new_head = nn.Linear(in_features, self.num_classes, bias=False)
            
            # === 修改：初始化权重为 0 ===
            nn.init.constant_(new_head.weight, 0)
            
            model.head = new_head
            print(f"    [Surgery] Head rebuilt: {in_features} -> {self.num_classes} classes (Bias Free). Weights initialized to 0.")

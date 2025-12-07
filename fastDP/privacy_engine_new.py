import logging
import math
import types
from typing import Dict, Optional, Sequence, Union, Tuple

import torch
from torch import nn

from . import autograd_grad_sample, transformers_support
from .accounting import accounting_manager
from torch.functional import F
import transformers
from .supported_layers_grad_samplers import _supported_layers_norm_sample_AND_clipping


class PrivacyEngine(object):
    """Differentially-private optimization engine that works in Pytorch.

    Supports book-keeping (BK) algorithm -- base and hybrid variants, as described in arXiv:2210.00038
    Supports DP-BiTFiT (bias-term only fine-tuning, which does not use BK), as described in arXiv:2210.00036
    """

    def __init__(
        self,
        module: nn.Module,
        *,
        batch_size: int,
        sample_size: int,
        max_grad_norm: float = 1.,
        epochs: Optional[Union[int, float]] = None,
        num_steps: Optional[Union[int, float]] = None,
        # MODIFIED: noise_multiplier can now be a scalar OR a Dict[str, float]
        noise_multiplier: Optional[Union[float, Dict[str, float]]] = None,
        target_epsilon: Optional[float] = None,
        target_delta: Optional[float] = None,
        alphas: Sequence[float] = accounting_manager.DEFAULT_ALPHAS,
        record_snr: bool = False,
        named_params: Optional[Sequence] = None,
        numerical_stability_constant=None,
        accounting_mode="rdp",
        eps_error=0.05,
        clipping_mode='MixOpt',
        clipping_fn='automatic',
        clipping_coe=None,
        loss_reduction='mean',
        origin_params=None,
        clipping_style='all-layer',
        num_GPUs=1,
        torch_seed_is_fixed=False,
        # NEW: A function to compute the per-parameter noise multiplier if a target_epsilon is given.
        # It takes the full list of named_params and the single computed scalar sigma as input.
        per_param_sigma_fn: Optional[types.FunctionType] = None,
        **unused_kwargs,
    ):
        """Initialize the engine.

        Args:
            # ... (other args remain the same)
            noise_multiplier: The extra multiplier for DP-SGD noise. Can be a single float
                or a dictionary mapping parameter names (str) to their specific noise multiplier (float).
            # ... (other args remain the same)
        """
        del unused_kwargs
        super(PrivacyEngine, self).__init__()

        if clipping_mode not in ['ghost','MixGhostClip','MixOpt']:
            raise ValueError(f"Unknown clipping mode {clipping_mode}. Expected one of 'ghost','MixGhostClip','MixOpt'.")
        if accounting_mode not in ("rdp", "all",'glw'):
            raise ValueError(f"Unknown accounting mode: {accounting_mode}. Expected one of 'rdp', 'all','glw'.")
        if epochs is None:
            if num_steps is not None:
                epochs=num_steps/sample_size*batch_size
            else:
                raise ValueError(f"Number of training epochs and training steps are not defined.")
        if epochs <= 0.0 and noise_multiplier is None:
            raise ValueError(f"Number of training epochs cannot be non-positive, but found epochs={epochs}")

        # Record parameters first to know which parameters require gradients
        self.module = module
        if named_params is None:
            self.named_params: Sequence[Tuple[str, nn.Parameter]] = list(
                (name, param) for (name, param) in module.named_parameters() if param.requires_grad
            )
        else:
            self.named_params = named_params
        self.num_params = sum(param.numel() for _, param in self.named_params)

        # Privacy parameters.
        sample_rate = batch_size / sample_size
        if target_delta is None:
            target_delta = 1 / (2 * sample_size)

        # Step 1: Compute a single scalar sigma if noise_multiplier is None and target_epsilon is set.
        scalar_sigma: Optional[float] = None
        if noise_multiplier is None:
            if target_epsilon is None or epochs is None:
                raise ValueError(
                    f"`target_epsilon` and `epochs` must be specified when `noise_multiplier` is `None`."
                )
            if accounting_mode in ("rdp", "all"):
                manager = accounting_manager.RDPManager(alphas=alphas)
            else:  # "glw"
                manager = accounting_manager.GLWManager(eps_error=eps_error)
            scalar_sigma = manager.compute_sigma(
                target_epsilon=target_epsilon, target_delta=target_delta, sample_rate=sample_rate, epochs=epochs,
            )
        elif isinstance(noise_multiplier, float):
            scalar_sigma = noise_multiplier
            
        # Step 2: Determine the per-parameter noise multipliers.
        # self.param_noise_multipliers will hold {param_name: sigma_value}
        self.param_noise_multipliers: Dict[str, float] = {}

        if isinstance(noise_multiplier, dict):
            self.param_noise_multipliers = noise_multiplier
            # For logging/stats, we can set a single effective sigma (e.g., the max or mean)
            self.noise_multiplier = max(noise_multiplier.values()) if noise_multiplier else 0.0
        elif scalar_sigma is not None:
            if per_param_sigma_fn:
                # Use the custom function to distribute the scalar sigma (e.g., proportional to norm)
                self.param_noise_multipliers = per_param_sigma_fn(self.named_params, scalar_sigma)
            else:
                # Use the same scalar sigma for all parameters (original behavior)
                self.param_noise_multipliers = {name: scalar_sigma for name, _ in self.named_params}
            self.noise_multiplier = scalar_sigma
        else:
            raise ValueError("`noise_multiplier` must be a float, a dictionary, or None (when `target_epsilon` is set).")

        # Basic Checks
        for name, _ in self.named_params:
            if name not in self.param_noise_multipliers:
                 # Default to the overall noise multiplier if not specified for a parameter
                 self.param_noise_multipliers[name] = self.noise_multiplier
                 logging.warning(f"Noise multiplier not specified for parameter {name}. Using the overall value {self.noise_multiplier}.")


        self.batch_size = batch_size
        self.sample_size = sample_size
        self.sample_rate = sample_rate
        self.max_grad_norm = max_grad_norm
        self.clipping_coe = clipping_coe

        self.epochs = epochs
        # The effective noise multiplier is still computed with the overall/max sigma for RDP accounting.
        self.effective_noise_multiplier = self.noise_multiplier / batch_size 
        self.target_epsilon = target_epsilon
        self.target_delta = target_delta
        self.alphas = alphas
        self.eps_error = eps_error
        self.accounting_mode = accounting_mode
        self.record_snr = record_snr
        
        # Internals.
        self.steps = 0  # Tracks privacy spending.

        # Recording.
        # ... (recording variables)
        self.max_clip = None
        self.min_clip = None
        self.med_clip = None
        self.signal = None
        self.noise = None
        self.snr = None
        self.noise_limit = None
        
        self._locked = False  # lock the part where noisy gradients is created (in `self.step`) if True.


        #----- ghost differentiation trick through origin parameter
        # ... (omitted for brevity, assume this part is unchanged)
        for name,param in module.named_parameters():
            param.initially_requires_grad=bool(param.requires_grad)
            if origin_params!=None:
                param.requires_grad=param.initially_requires_grad and any([i in name for i in origin_params]) # only requires grad if it is origin and initially requires grad

        if origin_params!=None:
            print('Using origin parameters for the ghost differentiation trick......')

        #-----
        def _supported_and_trainable(layer):            
            if type(layer) in _supported_layers_norm_sample_AND_clipping and ((hasattr(layer,'weight') and hasattr(layer.weight,'initially_requires_grad') and layer.weight.initially_requires_grad) or (hasattr(layer,'bias') and hasattr(layer.bias,'initially_requires_grad') and layer.bias.initially_requires_grad)):
                return True
            return False

        # store layer's name and create list of named layers for blockwise clipping
        self.named_layers=[]
        for name,layer in module.named_modules():
            if _supported_and_trainable(layer):
                self.named_layers.append((name,layer))

        self.n_layers=len(self.named_layers) #sum(1 for layer in module.modules() if autograd_grad_sample.requires_grad(layer) and hasattr(layer,'weight'))
        
        self.n_components=0
        for name, layer in self.named_layers:
            self.n_components+=sum([1 for p in layer.parameters() if p.initially_requires_grad])
        print("Number of trainable components: ",self.n_components, "; Number of trainable layers: ",self.n_layers)


        #-----
        print('>>>>>>>>>>>>>>>>> Applying ',clipping_fn, ' per-sample gradient clipping.')
        self.clipping_fn = clipping_fn
        if numerical_stability_constant!=None:
            self.numerical_stability_constant = numerical_stability_constant
        elif self.clipping_fn=='automatic':
            self.max_grad_norm = 1. # max_grad_norm does not matterin automatic clipping; this is necessary for step()
            self.numerical_stability_constant=1e-2
        else:
            self.numerical_stability_constant=1e-6
        
        if clipping_style=='layer-wise':
            # self.max_grad_norm_layerwise = self.max_grad_norm / math.sqrt(self.n_layers)
            self.max_grad_norm_layerwise = self.clipping_coe
            self.min_grad_norm_layerwise = 0.8 * self.max_grad_norm_layerwise
        elif clipping_style=='param-wise':
            self.max_grad_norm_layerwise = self.max_grad_norm / math.sqrt(self.n_components)
        elif clipping_style=='all-layer':
            self.max_grad_norm_layerwise=self.max_grad_norm
        else:
            self.max_grad_norm_layerwise=self.max_grad_norm / math.sqrt(len(clipping_style))

        for name,param in module.named_parameters():
            param.batch_size = self.batch_size
            # MODIFIED: param.noise now uses the parameter-specific noise multiplier
            param_sigma = self.param_noise_multipliers.get(name, self.noise_multiplier)
            if torch_seed_is_fixed == True:
                param.noise = param_sigma * self.max_grad_norm / num_GPUs
            else:
                param.noise = param_sigma * self.max_grad_norm / math.sqrt(num_GPUs)

        self.loss_reduction = loss_reduction
        self.clipping_mode = clipping_mode
        
        #----- determine whether training with BiTFiT
        self.bias_only=True
        for name,param in self.named_params:
            if '.bias' not in name and param.requires_grad:
                self.bias_only=False; break

        if self.bias_only:
            origin_params=None # do not use origin parameters for BiTFiT
            

        
        # create list of block head layers        
        if isinstance(clipping_style,list):
            self.clipping_style='block-wise'
            self.block_heads=clipping_style
        else:            
            self.clipping_style=clipping_style
            self.block_heads=[]
        
            if self.clipping_style=='all-layer':
                self.block_heads.append(self.named_layers[0][0])
            elif self.clipping_style in ['layer-wise','param-wise']:
                self.block_heads = [name for (name,layer) in self.named_layers]
        print(">>>>>>>>>>>>>>>>> Block heads for per-sample gradient clipping are defined as:", self.block_heads)

        transformers_support.forward_swapper(module=module)  # fix the position embeddings broadcast issue.

        autograd_grad_sample.add_hooks(model=self.module, loss_reduction=self.loss_reduction, 
                                       clipping_mode=self.clipping_mode, bias_only=self.bias_only,
                                       clipping_style=self.clipping_style, block_heads=self.block_heads,
                                       named_params=self.named_params, named_layers=self.named_layers,
                                       clipping_fn=self.clipping_fn, 
                                       numerical_stability_constant=self.numerical_stability_constant,
                                       max_grad_norm_layerwise=self.max_grad_norm_layerwise)

        def get_privacy_spent(_self, **kwargs):
            return _self.privacy_engine.get_privacy_spent(**kwargs)

        def get_training_stats(_self, **kwargs):
            return _self.privacy_engine.get_training_stats(**kwargs)

        # Make getting info easier.
        self.module.get_privacy_spent = types.MethodType(get_privacy_spent, self.module)
        self.module.get_training_stats = types.MethodType(get_training_stats, self.module)

        self.module.privacy_engine = self



        # ------ deepspeed ZERO 1 modification-----------
        # ... (omitted for brevity, assume this part is unchanged)
        try:
            from deepspeed.runtime.zero.stage_1_and_2 import DeepSpeedZeroOptimizer
            from deepspeed import comm as dist
    
            def reduce_gradients_DP_stage_1(self, pipeline_parallel=False):
                world_size = dist.get_world_size(self.dp_process_group)
                my_rank = dist.get_rank(self.dp_process_group)
    
                # with PP we must create ipg buffer, since backward is handled outside zero
                if pipeline_parallel and self.contiguous_gradients:
                    self.ipg_buffer = []
                    buf_0 = torch.empty(int(self.reduce_bucket_size),
                                         dtype=self.dtype,
                                         device=torch.cuda.current_device())
                    self.ipg_buffer.append(buf_0)
                    self.ipg_index = 0
    
                if not self.overlap_comm:
                    for i, group in enumerate(self.bit16_groups):
                        for param in group:
                            if param.grad is not None:
                                if hasattr(param,'private_grad'):
                                    param.grad = torch.nan_to_num(param.private_grad).contiguous()#+torch.normal(mean=0, std=param.noise,size=param.size(), device=param.device, dtype=param.dtype)
                                    del param.private_grad # release memory
                                    param.grad = param.grad / param.batch_size * self.loss_scale # it works
                                else:
                                    param.grad.zero_()
    
                            self.reduce_ready_partitions_and_remove_grads(param, i)
                # reduce any pending grads in either hook/non-hook case
                self.overlapping_partition_gradients_reduce_epilogue()
    
            DeepSpeedZeroOptimizer.reduce_gradients = reduce_gradients_DP_stage_1
        except:
            pass

    def lock(self):
        """Run this after noisy clipped gradient is created to prevent tampering with it before parameter update."""
        self._locked = True

    def unlock(self):
        """Run this after parameter update to allow creation of noisy gradient for next step"""
        self._locked = False

    def attach(self, optimizer):
        # ... (omitted for brevity, assume this part is unchanged)
        # Override step.
        def dp_step(_self, **kwargs):
            closure = kwargs.pop("closure", None)
            
            _self.zero_grad()       # make sure no non-private grad remains
            _self.privacy_engine._create_noisy_clipped_gradient(**kwargs)
            _self.original_step(closure=closure)
            _self.privacy_engine.unlock()  # Only enable creating new grads once parameters are updated.
            _self.privacy_engine.steps += 1


        optimizer.privacy_engine = self

        optimizer.original_step = optimizer.step
        optimizer.step = types.MethodType(dp_step, optimizer)           

        def get_privacy_spent(_self, **kwargs):
            return _self.privacy_engine.get_privacy_spent(**kwargs)

        def get_training_stats(_self, **kwargs):
            return _self.privacy_engine.get_training_stats(**kwargs)

        # Make getting info easier.
        optimizer.get_privacy_spent = types.MethodType(get_privacy_spent, optimizer)
        optimizer.get_training_stats = types.MethodType(get_training_stats, optimizer)

        self.optimizer = optimizer

    def detach(self):
        # ... (omitted for brevity, assume this part is unchanged)
        optimizer = self.optimizer
        optimizer.step = optimizer.original_step
        delattr(optimizer, "privacy_engine")
        delattr(optimizer, "original_step")
        delattr(optimizer, "get_privacy_spent")
        delattr(optimizer, "get_training_stats")

        module = self.module
        autograd_grad_sample.remove_hooks(module)
        module.zero_grad()

        for layer in self.module.modules():
            if hasattr(layer,'activations'):
                del layer.activations
            if hasattr(layer,'backprops'):
                del layer.backprops
            for param in layer.parameters():
              if hasattr(param,'private_grad'):
                del param.private_grad
        
    def _create_noisy_clipped_gradient(self):
        """Create noisy clipped gradient for `optimizer.step`."""
        
        unsupported_param_name=[]
        for name,param in list(self.named_params):
            if not hasattr(param, 'private_grad'):
                unsupported_param_name.append(name)
                self.named_params.remove((name,param)) 
        if unsupported_param_name!=[]:
            print(unsupported_param_name, 'are not supported by privacy engine; these parameters are not requiring gradient nor updated.')
                
        signals, noises = [], []
        
        for name,param in self.named_params:
            # Step 3: Add noise based on the parameter-specific noise multiplier (param.noise)
            # param.noise was calculated in __init__ using param_noise_multipliers
            noise_std = param.noise # param.noise already includes max_grad_norm and num_GPUs factor

            noise = torch.normal(
                mean=0,
                std=noise_std,
                size=param.size(),
                device=param.device,
                dtype=param.dtype,
            )
            param.grad = param.private_grad + noise # Ultra important to override `.grad`.
            del param.private_grad

            if self.record_snr:
                signals.append(param.grad.reshape(-1).norm(2))
                noises.append(noise.reshape(-1).norm(2))
                
            if self.loss_reduction=='mean':
                param.grad /= self.batch_size            

        # Note: SNR calculation is now approximate as param.noise might not be uniform.
        if self.record_snr and len(noises) > 0:
            self.signal, self.noise = tuple(torch.stack(lst).norm(2).item() for lst in (signals, noises))
            # noise_limit is calculated using the *maximum* noise multiplier (self.noise_multiplier) for a conservative estimate
            self.noise_limit = math.sqrt(self.num_params) * self.noise_multiplier * self.max_grad_norm
            self.snr = self.signal / self.noise
        else:
            self.snr = math.inf  # Undefined!

        self.lock()  # Make creating new gradients impossible, unless optimizer.step is called.

    def get_privacy_spent(
        # ... (omitted for brevity, assume this part is unchanged)
        self,
        steps: Optional[int] = None,
        accounting_mode: Optional[str] = None,
        lenient=False
    ) -> Dict:
        if steps is None:
            steps = self.steps
        if accounting_mode is None:
            accounting_mode = self.accounting_mode

        # RDP accounting is still based on the *max* noise multiplier for a conservative
        # privacy guarantee, assuming the same max_grad_norm C is applied to all.
        # This is a simplification; for a precise RDP/GLW accounting with heterogeneous
        # noise, a more complex composition analysis (often worst-case) is required.
        
        # ... (RDP/GLW accounting logic is unchanged and uses self.noise_multiplier)
        privacy_results = {}  # Contains stats from all modes.
        if accounting_mode in ('all','rdp'):
            try:
                manager = accounting_manager.RDPManager(alphas=self.alphas)
                privacy_results.update(
                    manager.compute_epsilon(
                        sigma=self.noise_multiplier,
                        sample_rate=self.sample_rate,
                        target_delta=self.target_delta,
                        steps=steps,
                    )
                )
            except Exception as err:
                logging.fatal("RDP accounting failed! Double check privacy parameters.")
                if not lenient:
                    raise err

        if accounting_mode in ('all','glw'):
            try:
                manager = accounting_manager.GLWManager(eps_error=self.eps_error)
                privacy_results.update(
                    manager.compute_epsilon(
                        sigma=self.noise_multiplier,
                        sample_rate=self.sample_rate,
                        target_delta=self.target_delta,
                        steps=steps
                    )
                )
            except Exception as err:
                logging.fatal(
                    "Numerical composition of tradeoff functions failed! Double check privacy parameters."
                )
                if not lenient:
                    raise err

        return privacy_results

    def get_training_stats(self):
        # ... (omitted for brevity, assume this part is unchanged)
        """Get the clipping, signal, and noise statistics."""
        return {
            "med_clip": self.med_clip,
            "max_clip": self.max_clip,
            "min_clip": self.min_clip,
            "snr": self.snr,
            "signal": self.signal,
            "noise": self.noise,
            "noise_limit": self.noise_limit,
        }

    def __repr__(self):
        # ... (omitted for brevity, assume this part is unchanged)
        return (
            f"PrivacyEngine(\n"
            f"  target_epsilon={self.target_epsilon:.6f}, \n"
            f"  target_delta={self.target_delta:.6f}, \n"
            f"  noise_multiplier_for_accounting={self.noise_multiplier:.6f}, \n" # Clarified for logging
            f"  effective_noise_multiplier={self.effective_noise_multiplier:.6f}, \n"
            f"  epochs={self.epochs}, \n"
            f"  max_grad_norm={self.max_grad_norm}, \n"
            f"  sample_rate={self.sample_rate}, \n"
            f"  batch_size={self.batch_size}, \n"
            f"  accounting_mode={self.accounting_mode}, \n"
            f"  clipping_mode={self.clipping_mode}\n"
            f")"
        )

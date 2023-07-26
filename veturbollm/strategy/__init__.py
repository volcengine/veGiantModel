import torch

from veturbollm.global_vars import get_args
from veturbollm.optim.base import get_optimizer_with_scheduler
from veturbollm.utils.dtype import get_torch_dtype
from veturbollm.utils.operations import convert_outputs_to_fp32

from .ddp import DDPStrategy
from .fsdp import FSDPStrategy


def prepare_distributed_strategy(model):
    args = get_args()

    if args.distributed.strategy == "ddp":
        strategy = DDPStrategy()
        model.model, _ = strategy.setup_model_and_optimizer(model.model, None)
    elif args.distributed.strategy == "fsdp":
        strategy = FSDPStrategy(
            precision=args.model.precision,
            forward_prefetch=args.distributed.fsdp_strategy_config.forward_prefetch,
            limit_all_gathers=args.distributed.fsdp_strategy_config.limit_all_gathers,
            sync_module_states=args.distributed.fsdp_strategy_config.sync_module_states,
            activation_checkpointing=args.distributed.fsdp_strategy_config.activation_checkpointing,
            use_orig_params=args.distributed.fsdp_strategy_config.use_orig_params,
        )
        model.model, _ = strategy.setup_model_and_optimizer(model.model, None)
    else:
        raise NotImplementedError(f"strategy {args.distributed.strategy} is not implemented")

    optimizer, lr_scheduler = get_optimizer_with_scheduler(model)

    dtype = get_torch_dtype(args.model.precision)
    if args.model.enable_native_amp:
        model._original_forward = model.forward
        if dtype == torch.float16:
            model.forward = torch.cuda.amp.autocast(dtype=torch.float16)(model.forward)
        elif dtype == torch.bfloat16:
            model.forward = torch.autocast(device_type="cuda", dtype=torch.bfloat16)(model.forward)
        else:
            model.forward = torch.cuda.amp.autocast()(model.forward)
        model.forward = convert_outputs_to_fp32(model.forward)
    elif args.model.mixed_precision == "fp8":
        import transformer_engine.common.recipe as te_recipe
        from transformer_engine.pytorch import fp8_autocast

        from veturbollm.utils.transformer_engine import convert_model, has_transformer_engine_layers

        if not has_transformer_engine_layers(model):
            with torch.no_grad():
                convert_model(model)
            model._converted_to_transformer_engine = True
        model._original_forward = model.forward

        kwargs = args.model.fp8_recipe_handler.to_kwargs()
        if "fp8_format" in kwargs:
            kwargs["fp8_format"] = getattr(te_recipe.Format, kwargs["fp8_format"])
        fp8_recipe = te_recipe.DelayedScaling(**kwargs)
        fp8_enabled = torch.cuda.get_device_capability()[0] >= 9
        if not fp8_enabled:
            print(
                f"The current device has compute capability of {torch.cuda.get_device_capability()} which is "
                "insufficient for FP8 mixed precision training (requires a GPU Hopper or higher, compute "
                "capability of 9 or higher). Will use FP16 instead."
            )
        model.forward = fp8_autocast(enabled=fp8_enabled, fp8_recipe=fp8_recipe)(model.forward)

    # torch.compile should be called last.
    if args.model.enable_dynamo:
        if not hasattr(torch, "compile"):
            raise ValueError("Using torch.compile requires PyTorch 2.0 or higher.")
        model = torch.compile(model)
    model.model.device = torch.device("cuda")
    model.device = torch.device("cuda")
    return model, optimizer, lr_scheduler

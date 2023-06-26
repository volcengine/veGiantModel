from typing import Any, Dict, List, Optional, Tuple, Union

from pydantic import BaseModel


class TokenizerConfig(BaseModel):
    pretrained_model_name_or_path: str = "gpt2"
    use_slow_tokenizer: bool = False


class ModelConfig(BaseModel):
    pretrained_model_name_or_path: str = "gpt2"
    trust_remote_code: bool = True
    use_auth_token: bool = False
    pretrained: bool = False
    init_device: str = "cpu"  # or meta
    enable_native_amp: bool = True
    enable_dynamo: bool = True
    enable_flash_attn: bool = False
    precision: str = "amp_bf16"
    mixed_precision: str = "bf16"
    fp8_recipe_handler: Dict[str, Any] = {}
    config_overrides: Dict[str, Any] = {}


class DistributedConfig(BaseModel):
    strategy: str = "ddp"  # ddp, fsdp
    data_parallel_size: int = 1
    tensor_model_parallel_size: int = 1
    pipeline_model_parallel_size: int = 1


class LearningRateConfig(BaseModel):
    lr: float = 5e-5
    lr_decay_style: str = "linear"
    lr_decay_iters: Optional[int] = None
    lr_decay_samples: Optional[int] = None
    lr_warmup_iters: int = 0
    lr_warmup_samples: int = 0


class DatasetConfig(BaseModel):
    dataset_name: Optional[str] = "wikitext"
    dataset_config_name: Optional[str] = "wikitext-2-raw-v1"
    train_file: Optional[str] = None
    validation_file: Optional[str] = None
    block_size: int = 1024
    preprocessing_num_workers: int = 8
    overwrite_cache: bool = False


class RegularizationConfig(BaseModel):
    weight_decay: float = 0.01


class CheckpointingConfig(BaseModel):
    save: bool = False
    save_interval: Optional[int] = None
    load: bool = False


class LoggingConfig(BaseModel):
    timing_log_level: int = 0
    timing_log_option: str = "minmax"
    log_params_norm: Optional[Union[str, List[str]]] = None
    log_num_zeros_in_grad: Optional[bool] = None
    no_barrier_with_level_1_timing: Optional[bool] = None
    tensorboard_log_interval: int = 1
    tensorboard_queue_size: int = 1000
    log_timers_to_tensorboard: bool = True
    log_batch_size_to_tensorboard: bool = True
    log_learning_rate_to_tensorboard: bool = True
    log_loss_scale_to_tensorboard: bool = True
    log_validation_ppl_to_tensorboard: bool = True
    log_memory_to_tensorboard: bool = True
    log_world_size_to_tensorboard: bool = True


class TrainingConfig(BaseModel):
    micro_batch_size: int = 8
    global_batch_size: Optional[int] = None
    gradient_accumulation_steps: int = 1
    rampup_batch_size: Optional[int] = None
    train_iters: int = 10000
    train_epochs: Optional[int] = None
    train_samples: Optional[int] = None
    lr_warmup_iters: Optional[int] = None
    log_interval: int = 1
    tensorboard_dir: str = "./tensorboard_log"
    tensorboard_queue_size: int = 10


class TaskConfig(BaseModel):
    seed: int = 42
    rank: int = 0
    local_rank: int = 0
    world_size: int = 1
    consumed_train_samples: int = 0

    tokenizer: TokenizerConfig = TokenizerConfig()
    model: ModelConfig = ModelConfig()
    dataset: DatasetConfig = DatasetConfig()
    distributed: DistributedConfig = DistributedConfig()
    training: TrainingConfig = TrainingConfig()
    learning_rate: LearningRateConfig = LearningRateConfig()
    regularization: RegularizationConfig = RegularizationConfig()
    checkpointing: CheckpointingConfig = CheckpointingConfig()
    logging: LoggingConfig = LoggingConfig()

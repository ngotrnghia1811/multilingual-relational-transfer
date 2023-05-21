import copy
import json
import yaml
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any


@dataclass
class ModelConfig:
    bert_model_name: str = "xlm-roberta-base"
    bert_cache_dir: Optional[str] = None
    bert_dropout: float = 0.1
    hidden_dim: int = 256
    num_event_types: int = 17
    num_relation_types: int = 37
    num_entity_types: int = 5
    multi_piece_strategy: str = "first"


@dataclass
class TrainConfig:
    bert_learning_rate: float = 1e-5
    learning_rate: float = 1e-3
    bert_weight_decay: float = 1e-5
    weight_decay: float = 1e-3
    batch_size: int = 16
    eval_batch_size: int = 16
    max_epoch: int = 50
    warmup_epoch: int = 5
    grad_clipping: float = 5.0
    accumulate_step: int = 1
    seed: int = 42
    log_step: int = 100
    eval_step: int = 500


@dataclass
class DataConfig:
    data_dir: str = "data/"
    task: str = "event_detection"
    source_languages: List[str] = field(default_factory=list)
    target_languages: List[str] = field(default_factory=list)
    all_languages: List[str] = field(default_factory=list)
    max_length: int = 128
    use_gpu: bool = True
    gpu_device: int = 0


@dataclass
class GrDAConfig:
    lambda_gan: float = 0.1
    lr_e: float = 1e-4
    lr_d: float = 1e-4
    lr_g: float = 1e-4
    z_dim: int = 64
    hidden_dim: int = 128
    sample_v: int = 4
    sample_v_g: int = 4
    num_epoch_pretrain_g: int = 10


@dataclass
class ZSRLConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    data: DataConfig = field(default_factory=DataConfig)
    grda: Optional[GrDAConfig] = None
    log_dir: str = "logs/"
    output_dir: str = "checkpoints/"
    transfer_mode: str = "zscls"

    @classmethod
    def from_yaml(cls, path: str) -> "ZSRLConfig":
        with open(path, "r", encoding="utf-8") as f:
            d = yaml.safe_load(f)
        return cls._from_dict(d)

    @classmethod
    def _from_dict(cls, d: Dict[str, Any]) -> "ZSRLConfig":
        model = ModelConfig(**d.get("model", {}))
        train = TrainConfig(**d.get("train", {}))
        data = DataConfig(**d.get("data", {}))
        grda = GrDAConfig(**d["grda"]) if d.get("grda") else None
        return cls(
            model=model,
            train=train,
            data=data,
            grda=grda,
            log_dir=d.get("log_dir", "logs/"),
            output_dir=d.get("output_dir", "checkpoints/"),
            transfer_mode=d.get("transfer_mode", "zscls"),
        )

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        return d

    def save(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)

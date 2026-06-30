"""Configuration management for IndicF5 Neo"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import torch

@dataclass
class AudioConfig:
    """Audio processing configuration"""
    sample_rate: int = 24000
    n_mel_channels: int = 100
    hop_length: int = 256
    win_length: int = 1024
    n_fft: int = 1024
    mel_spec_type: str = "vocos"
    target_rms: float = 0.1


@dataclass
class InferenceConfig:
    """Inference generation configuration"""
    nfe_steps: int = 16
    cfg_strength: float = 1.5
    sway_sampling_coef: float = -1.0
    speed: float = 1.0
    cross_fade_duration: float = 0.15
    fix_duration: Optional[int] = None
    ode_method: str = "euler"


@dataclass
class ModelConfig:
    """Model loading configuration"""
    model_repo: str = os.getenv("INDICF5_MODEL_PATH", "ai4bharat/IndicF5")
    vocoder_repo: str = os.getenv("INDICF5_VOCODER_PATH", "charactr/vocos-mel-24khz")
    device: str = "auto"  # auto, cuda, cpu
    dtype: str = "float32"  # float32, float16
    use_mixed_precision: bool = True
    torch_compile: bool = False
    compile_mode: str = "reduce-overhead"
    enable_tf32: bool = True
    cudnn_benchmark: bool = True
    warmup_on_load: bool = False
    enable_cache: bool = True
    
    def __post_init__(self):
        """Check if local model exists and use it if so"""
        local_model = Path("models/IndicF5")
        if local_model.exists() and any(local_model.iterdir()) and self.model_repo == "ai4bharat/IndicF5":
            self.model_repo = str(local_model.absolute())
    
    def get_device(self) -> str:
        """Get appropriate device"""
        if self.device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return self.device


@dataclass
class UIConfig:
    """UI and server configuration"""
    host: str = "127.0.0.1"
    port: int = 7860
    share: bool = False
    theme: str = "soft"
    debug: bool = False
    max_queue_size: int = 10


@dataclass
class PathsConfig:
    """Project paths configuration"""
    root_dir: Path
    profiles_dir: Path
    outputs_dir: Path
    cache_dir: Path
    logs_dir: Path
    
    def __post_init__(self):
        """Create directories if they don't exist"""
        for dir_path in [self.profiles_dir, self.outputs_dir, self.cache_dir, self.logs_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)


class AppConfig:
    """Main application configuration"""
    
    def __init__(self, root_dir: Optional[Path] = None):
        if root_dir is None:
            root_dir = Path(__file__).resolve().parent.parent  # IndicF5_Neo root
        
        self.root = Path(root_dir)
        
        # Initialize all configs
        self.audio = AudioConfig()
        self.inference = InferenceConfig()
        self.model = ModelConfig()
        self.ui = UIConfig()
        self.paths = PathsConfig(
            root_dir=self.root,
            profiles_dir=self.root / "profiles",
            outputs_dir=self.root / "outputs",
            cache_dir=self.root / ".cache",
            logs_dir=self.root / "logs"
        )
    
    def get_device(self) -> str:
        """Get the device to use"""
        return self.model.get_device()
    
    def to_dict(self) -> dict:
        """Convert config to dictionary"""
        return {
            "audio": self.audio.__dict__,
            "inference": self.inference.__dict__,
            "model": {
                k: v for k, v in self.model.__dict__.items() 
                if k != "dtype"
            },
            "ui": self.ui.__dict__,
            "paths": {k: str(v) for k, v in self.paths.__dict__.items()},
        }


# Singleton instance
_config = None

def get_config() -> AppConfig:
    """Get or create the app config singleton"""
    global _config
    if _config is None:
        _config = AppConfig()
    return _config

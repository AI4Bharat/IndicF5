"""IndicF5 Neo - Fast and Scalable Text-to-Speech"""

__version__ = "1.0.0"
__author__ = "Your Team"
__description__ = "High-quality text-to-speech for Indic languages"

from src.config import get_config, AppConfig
from src.utils.logger import get_logger, LoggerSetup
from src.utils.device_manager import DeviceManager
from src.inference.engine import get_inference_engine, IndicF5InferenceEngine
from src.data_management.profile_manager import VoiceProfileManager

__all__ = [
    "get_config",
    "AppConfig",
    "get_logger",
    "LoggerSetup", 
    "DeviceManager",
    "get_inference_engine",
    "IndicF5InferenceEngine",
    "VoiceProfileManager",
]

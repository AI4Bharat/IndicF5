"""Device management for PyTorch models"""

import torch
from typing import Optional
from src.utils.logger import get_logger

logger = get_logger(__name__)


class DeviceManager:
    """Manage device assignment and model movement"""
    
    def __init__(self, initial_device: Optional[str] = None):
        """
        Initialize device manager
        
        Args:
            initial_device: Device to use ('cuda', 'cpu', etc.)
        """
        if initial_device and initial_device in ['cuda', 'cpu']:
            self.device = torch.device(initial_device)
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        
        logger.info(f"Device initialized to: {self.device}")
        self._log_device_info()

    def _log_device_info(self) -> None:
        """Log device information"""
        if torch.cuda.is_available():
            logger.info(f"CUDA Version: {torch.version.cuda}")
            logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
            logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        else:
            logger.info("CUDA not available, using CPU")

    def set_device(self, device_str: str) -> torch.device:
        """
        Set the device
        
        Args:
            device_str: Device string ('cuda' or 'cpu')
            
        Returns:
            The device object
        """
        if device_str not in ['cuda', 'cpu']:
            logger.warning(f"Invalid device {device_str}, keeping {self.device}")
            return self.device
        
        if device_str == 'cuda' and not torch.cuda.is_available():
            logger.warning("CUDA requested but not available, using CPU")
            self.device = torch.device('cpu')
        else:
            self.device = torch.device(device_str)
        
        logger.info(f"Device changed to: {self.device}")
        return self.device

    def move_to_device(self, model):
        """
        Move model to device
        
        Args:
            model: PyTorch model
            
        Returns:
            Model on the device
        """
        if model is not None:
            try:
                return model.to(self.device)
            except Exception as e:
                logger.error(f"Error moving model to device: {str(e)}")
                raise
        return None

    def get_current_device(self) -> torch.device:
        """Get the current device"""
        return self.device

    def clear_cache(self) -> None:
        """Clear GPU cache"""
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.info("GPU cache cleared")
        except Exception as e:
            logger.warning(f"Error clearing cache: {str(e)}")

    def get_device_string(self) -> str:
        """Get device as string"""
        return str(self.device)

    def is_cuda_available(self) -> bool:
        """Check if CUDA is available"""
        return torch.cuda.is_available()
    
    def get_gpu_memory_info(self) -> dict:
        """Get GPU memory information"""
        if not torch.cuda.is_available():
            return {"status": "CUDA not available"}
        
        return {
            "total_memory": torch.cuda.get_device_properties(0).total_memory,
            "allocated_memory": torch.cuda.memory_allocated(),
            "reserved_memory": torch.cuda.memory_reserved(),
            "free_memory": torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()
        }


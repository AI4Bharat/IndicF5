"""Logging setup for IndicF5 Neo"""

import logging
import logging.handlers
from pathlib import Path
from typing import Optional


class LoggerSetup:
    """Configure application-wide logging"""
    
    _logger = None
    
    @classmethod
    def setup(cls, log_dir: Optional[Path] = None, level: int = logging.INFO) -> logging.Logger:
        """
        Setup logger with file and console handlers
        
        Args:
            log_dir: Directory to save log files
            level: Logging level
            
        Returns:
            Configured logger instance
        """
        if cls._logger is not None:
            return cls._logger
        
        # Create logger
        logger = logging.getLogger("IndicF5Neo")
        logger.setLevel(level)
        logger.propagate = False
        
        # Remove existing handlers
        logger.handlers.clear()
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_formatter = logging.Formatter(
            "[%(asctime)s] %(levelname)-8s [%(name)s:%(funcName)s:%(lineno)d] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
        
        # File handler (if log_dir provided)
        if log_dir:
            log_dir = Path(log_dir)
            log_dir.mkdir(parents=True, exist_ok=True)
            
            file_handler = logging.handlers.RotatingFileHandler(
                log_dir / "indicf5_neo.log",
                maxBytes=10 * 1024 * 1024,  # 10MB
                backupCount=5
            )
            file_handler.setLevel(logging.DEBUG)
            file_formatter = logging.Formatter(
                "[%(asctime)s] %(levelname)-8s [%(name)s:%(funcName)s:%(lineno)d] %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S"
            )
            file_handler.setFormatter(file_formatter)
            logger.addHandler(file_handler)
        
        cls._logger = logger
        return logger


def get_logger(name: str = "IndicF5Neo") -> logging.Logger:
    """Get a logger instance"""
    if not name or name == "IndicF5Neo":
        return logging.getLogger("IndicF5Neo")
    if name.startswith("IndicF5Neo"):
        return logging.getLogger(name)
    normalized = name.replace("__main__", "launch")
    return logging.getLogger(f"IndicF5Neo.{normalized}")

"""
Logging Utilities
Configures and provides logging functionality using loguru.
"""

import sys
from pathlib import Path
from typing import Optional

from loguru import logger

from src.utils.config import get_config


class LoggerManager:
    """Manages application logging."""
    
    _initialized = False
    
    @classmethod
    def setup_logger(cls, config_override: Optional[dict] = None) -> None:
        """Setup logger with configuration."""
        if cls._initialized:
            return
        
        config = get_config()
        log_config = config.logging
        
        if config_override:
            for key, value in config_override.items():
                setattr(log_config, key, value)
        
        # Remove default handler
        logger.remove()
        
        # Add console handler
        logger.add(
            sys.stdout,
            format=log_config.format,
            level=log_config.level,
            colorize=True
        )
        
        # Add file handler
        log_dir = Path(log_config.log_directory)
        log_dir.mkdir(parents=True, exist_ok=True)
        
        log_file = log_dir / log_config.log_file
        
        logger.add(
            str(log_file),
            format=log_config.format,
            level=log_config.level,
            rotation=log_config.rotation,
            retention=log_config.retention,
            compression="zip"
        )
        
        cls._initialized = True
        logger.info(f"Logger initialized: {log_config.level} level")
    
    @classmethod
    def get_logger(cls):
        """Get logger instance."""
        if not cls._initialized:
            cls.setup_logger()
        return logger


def get_logger():
    """Get logger instance."""
    return LoggerManager.get_logger()

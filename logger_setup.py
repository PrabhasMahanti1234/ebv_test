"""
logger_setup.py - Centralized Logging Configuration

Provides a reusable setup_logger() function that creates named loggers
with both console and rotating file handlers.
"""

import logging
import os
from logging.handlers import RotatingFileHandler


def setup_logger(name: str, log_file: str, level=logging.INFO,
                 max_bytes: int = 10 * 1024 * 1024, backup_count: int = 5) -> logging.Logger:
    """
    Create and configure a named logger with console + rotating file output.

    Args:
        name:         Logger name (e.g. 'database', 'extraction')
        log_file:     Path to the log file (e.g. 'logs/database.log')
        level:        Logging level (default: INFO)
        max_bytes:    Max log file size before rotation (default: 10 MB)
        backup_count: Number of backup files to keep (default: 5)

    Returns:
        Configured logging.Logger instance
    """
    logger = logging.getLogger(name)

    # Avoid adding duplicate handlers if logger already exists
    if logger.handlers:
        return logger

    logger.setLevel(level)

    formatter = logging.Formatter(
        '%(asctime)s,%(msecs)03d [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (rotating)
    if log_file:
        log_dir = os.path.dirname(log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        try:
            file_handler = RotatingFileHandler(
                log_file, maxBytes=max_bytes, backupCount=backup_count, encoding='utf-8'
            )
            file_handler.setLevel(level)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        except Exception as e:
            logger.warning(f"Could not create log file '{log_file}': {e}")

    return logger

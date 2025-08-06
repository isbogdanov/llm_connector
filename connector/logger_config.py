import logging
import os
from logging.handlers import RotatingFileHandler


def setup_logging():
    """Set up a robust, configurable logging system."""
    # Get the absolute path to the directory of this file (logger_config.py)
    # This ensures the log directory is created relative to the submodule's location
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Go up one level to the root of the llm_connector submodule
    submodule_root = os.path.dirname(current_dir)

    # Define the log directory path relative to the submodule's root
    log_dir = os.path.join(submodule_root, "logs")

    # Create logs directory if it doesn't exist
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Configure the logger
    log_file = os.path.join(log_dir, "llm_connector.log")
    logger = logging.getLogger("LLMConnector")
    logger.setLevel(logging.INFO)

    # Prevent duplicate handlers if this is called multiple times
    if logger.hasHandlers():
        logger.handlers.clear()

    # Create a rotating file handler
    # 1MB per file, keeping 5 backup files
    handler = RotatingFileHandler(log_file, maxBytes=1 * 1024 * 1024, backupCount=5)

    # Create a formatter and set it for the handler
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)

    # Add the handler to the logger
    logger.addHandler(handler)

    # Do not propagate to the root logger to avoid console output
    logger.propagate = False

    return logger


# Set up the logger on import
logger = setup_logging()

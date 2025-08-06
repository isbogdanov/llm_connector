import logging
import os
import datetime
from logging.handlers import RotatingFileHandler


def setup_timestamped_logging():
    """Set up a robust, configurable logging system with a timestamped file."""
    # Get the absolute path to the directory of this file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    submodule_root = os.path.dirname(current_dir)
    log_dir = os.path.join(submodule_root, "logs")

    # Create logs directory if it doesn't exist
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Create a unique, timestamped log file name
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file = os.path.join(log_dir, f"connector_{timestamp}.log")

    # Configure the logger
    logger = logging.getLogger("LLMConnector")
    logger.setLevel(logging.INFO)

    # Prevent duplicate handlers
    if logger.hasHandlers():
        logger.handlers.clear()

    # Create a file handler (not rotating, as each run has its own file)
    handler = logging.FileHandler(log_file)

    # Create a formatter and set it for the handler
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)

    # Add the handler to the logger
    logger.addHandler(handler)
    logger.propagate = False

    logger.info(f"Logging initialized to file: {log_file}")

    return logger

# Copyright 2026 Igor Bogdanov
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os
import datetime
import yaml

from .path_resolver import CONNECTOR_HOME, CONF_DIR, LOGS_DIR

def setup_timestamped_logging(level=logging.INFO):
    """Set up a robust, configurable logging system sourced from logs.yaml."""
    
    yaml_path = os.path.join(CONF_DIR, "logs.yaml")
    config = {}
    if os.path.exists(yaml_path):
        try:
            with open(yaml_path, "r") as f:
                config = yaml.safe_load(f) or {}
        except Exception as e:
            print(f"Failed to load logs.yaml: {e}")

    # Use the resolved LOGS_DIR from path_resolver (respects CONNECTOR_HOME)
    log_dir = LOGS_DIR

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    prefix = config.get("filename_prefix", "connector")
    date_fmt = config.get("date_format", "%Y-%m-%d_%H-%M-%S")
    timestamp = datetime.datetime.now().strftime(date_fmt)
    log_file = os.path.join(log_dir, f"{prefix}_{timestamp}.log")

    logger = logging.getLogger("LLMConnector")
    
    yaml_level_str = config.get("level", "INFO").upper()
    yaml_level = getattr(logging, yaml_level_str, logging.INFO)
    
    # Priority: Function Args (e.g. CLI debug flags) > YAML Configs > Default INFO
    final_level = level if level != logging.INFO else yaml_level
    logger.setLevel(final_level)

    if logger.hasHandlers():
        logger.handlers.clear()

    handler = logging.FileHandler(log_file)

    fmt_str = config.get("format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    formatter = logging.Formatter(fmt_str)
    handler.setFormatter(formatter)

    logger.addHandler(handler)
    logger.propagate = False

    logger.info(f"Logging dynamically initialized to file: {log_file}")

    return logger
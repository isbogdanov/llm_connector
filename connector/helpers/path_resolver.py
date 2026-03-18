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

"""
Central path resolver for the LLM Connector package.

Resolves CONNECTOR_HOME, CONF_DIR, and LOGS_DIR using a strict priority chain:
  1. LLM_CONNECTOR_HOME environment variable (explicit override)
  2. os.getcwd()/llm-connector (auto-discovery for pip-installed usage)
  3. Package's own directory (fallback for dev/submodule use)
"""

import os


def _resolve_connector_home():
    """Determine the root directory for configs and logs."""
    
    # Priority 1: Explicit environment variable
    env_home = os.environ.get("LLM_CONNECTOR_HOME")
    if env_home and os.path.isdir(env_home):
        return os.path.join(env_home, "conf"), os.path.join(env_home, "logs")
    
    # Priority 2: Auto-discover scaffolded folder in CWD
    cwd_home = os.path.join(os.getcwd(), "llm-connector")
    if os.path.isdir(cwd_home) and os.path.isdir(os.path.join(cwd_home, "conf")):
        return os.path.join(cwd_home, "conf"), os.path.join(cwd_home, "logs")
    
    # Priority 3: Fallback to package's own conf/ directory (dev/submodule mode)
    # This file lives in connector/helpers/, so one level up is connector/
    package_dir = os.path.dirname(os.path.dirname(__file__))
    package_conf = os.path.join(package_dir, "conf")
    # Logs go to the repo root (one level above connector/)
    repo_root = os.path.dirname(package_dir)
    return package_conf, os.path.join(repo_root, "logs")


# --- Resolved Paths ---
CONF_DIR, LOGS_DIR = _resolve_connector_home()

# For .env loading, CONNECTOR_HOME is the parent of CONF_DIR
CONNECTOR_HOME = os.path.dirname(CONF_DIR)

# --- Frozen template directory (always inside the installed package) ---
PACKAGE_CONF_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "conf")
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
  3. Package's own __file__ directory (fallback for dev/submodule use)
"""

import os
import importlib.resources

def _resolve_connector_home():
    """Determine the root directory for configs and logs."""
    
    # Priority 1: Explicit environment variable
    env_home = os.environ.get("LLM_CONNECTOR_HOME")
    if env_home and os.path.isdir(env_home):
        return env_home
    
    # Priority 2: Auto-discover scaffolded folder in CWD
    cwd_home = os.path.join(os.getcwd(), "llm-connector")
    if os.path.isdir(cwd_home) and os.path.isdir(os.path.join(cwd_home, "conf")):
        return cwd_home
    
    # Priority 3: Fallback to package's own directory (dev/submodule mode)
    # This file lives in connector/helpers/, so 2 levels up is the repo root
    package_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    return package_root


# --- Resolved Paths ---
CONNECTOR_HOME = _resolve_connector_home()
CONF_DIR = os.path.join(CONNECTOR_HOME, "conf")
LOGS_DIR = os.path.join(CONNECTOR_HOME, "logs")

# --- Frozen template directory (always inside the installed package) ---
PACKAGE_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
PACKAGE_CONF_DIR = os.path.join(PACKAGE_ROOT, "conf")
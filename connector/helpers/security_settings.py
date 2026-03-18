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

import os
import yaml

from .path_resolver import CONF_DIR

def _load_yaml(filename):
    path = os.path.join(CONF_DIR, filename)
    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                return yaml.safe_load(f) or {}
        except Exception as e:
            print(f"Error loading {filename}: {e}")
    return {}

# --- Security Variables ---
security_config = _load_yaml("security.yaml")

retry_config = security_config.get("retry", {})
RETRY_MAX = retry_config.get("max_retries", 3)
RETRY_BACKOFF = retry_config.get("backoff_factor", 0.5)
RETRY_STATUSES = retry_config.get("status_forcelist", [429, 500, 502, 503, 504])

resources_config = security_config.get("resources", {})
POOL_CONNECTIONS = resources_config.get("pool_connections", 10)
POOL_MAXSIZE = resources_config.get("pool_maxsize", 110)
RLIMIT_NOFILE = resources_config.get("rlimit_nofile", 4096)
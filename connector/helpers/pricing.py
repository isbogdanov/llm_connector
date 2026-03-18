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
import json
import glob
import logging
import requests
import datetime
from .llm_settings import config
from .path_resolver import CONF_DIR, LOGS_DIR

logger = logging.getLogger("LLMConnector")

class PricingManager:
    def __init__(self):
        self.model_pricing = {}
        self.dynamic_pricing = {}
        self.logs_dir = os.path.join(LOGS_DIR, "pricing")
        
        # Sequentially hydrate state from YAML overrides and JSON caches
        self._load_yaml_pricing()
        self._load_dynamic_pricing()

    def _load_yaml_pricing(self):
        """Parse native YAML constraints into the required Tuple format."""
        pricing_yaml = config.get("pricing", {})
        for provider, models in pricing_yaml.items():
            if isinstance(models, dict):
                for model_name, costs in models.items():
                    if isinstance(costs, list) and len(costs) >= 2:
                        self.model_pricing[(provider, model_name)] = (float(costs[0]), float(costs[1]))

    def _load_dynamic_pricing(self, strict_fallback=False):
        """Finds, validates, and parses the most recent OpenRouter proxy cache file."""
        if not os.path.exists(self.logs_dir):
            os.makedirs(self.logs_dir, exist_ok=True)
            
        pricing_files = glob.glob(os.path.join(self.logs_dir, "openrouter_pricing_*.json"))
        
        needs_update = False
        latest_file = None
        
        if not pricing_files:
            needs_update = True
        else:
            latest_file = max(pricing_files)
            try:
                filename = os.path.basename(latest_file)
                # openrouter_pricing_YYYY-MM-DD_HH-MM-SS.json
                date_str = filename.split('_')[2] 
                today_str = datetime.datetime.now().strftime("%Y-%m-%d")
                if date_str != today_str:
                    needs_update = True
            except Exception:
                needs_update = True
                
        if needs_update and not strict_fallback:
            logger.info("Pricing cache is missing or outdated. Triggering automatic background refresh...")
            self.update_openrouter_pricing()
            return
            
        if latest_file:
            try:
                with open(latest_file, 'r') as f:
                    self.dynamic_pricing = json.load(f)
            except Exception as e:
                logger.warning(f"Could not load {latest_file}: {e}")

    def update_openrouter_pricing(self):
        """Scrapes OpenRouter for live pricing and rigidly caches it."""
        logger.info("Fetching latest OpenRouter model pricing...")
        url = "https://openrouter.ai/api/v1/models"
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            data = response.json().get("data", [])
            
            pricing_dict = {}
            for model in data:
                model_id = model.get("id")
                pricing = model.get("pricing", {})
                try:
                    # Scale to 'per 1 million tokens' natively
                    prompt_cost = float(pricing.get("prompt", 0)) * 1_000_000
                    completion_cost = float(pricing.get("completion", 0)) * 1_000_000
                    pricing_dict[model_id] = (round(prompt_cost, 4), round(completion_cost, 4))
                except (ValueError, TypeError):
                    continue
                    
            os.makedirs(self.logs_dir, exist_ok=True)
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            output_file = os.path.join(self.logs_dir, f"openrouter_pricing_{timestamp}.json")
            
            with open(output_file, "w") as f:
                json.dump(pricing_dict, f, indent=4)
                
            logger.info(f"Successfully cached dynamic API costs for {len(pricing_dict)} models into {output_file}")
            
            # Hot-swap memory state and prevent infinite recursion loops
            self._load_dynamic_pricing(strict_fallback=True)
            
        except Exception as e:
            logger.error(f"Error fetching OpenRouter pricing: {e}")

    def get_model_pricing(self, provider: str, model: str) -> tuple:
        """Returns (prompt_cost_per_1m, completion_cost_per_1m)."""
        # 1. Evaluate Explicit YAML Defaults
        if (provider, model) in self.model_pricing:
            return self.model_pricing[(provider, model)]
        
        # 2. Evaluate Dynamic Network Caches
        lookup_id = f"{provider}/{model}" if provider != "openrouter" else model
        
        if lookup_id in self.dynamic_pricing:
            return self.dynamic_pricing[lookup_id]

        # 3. Normalize version dashes to dots (e.g. claude-haiku-4-5 -> claude-haiku-4.5)
        #    Anthropic native API uses dashes, OpenRouter pricing uses dots
        normalized = self._normalize_model_version(lookup_id)
        if normalized != lookup_id and normalized in self.dynamic_pricing:
            return self.dynamic_pricing[normalized]
            
        return (0.0, 0.0)

    def has_model_pricing(self, provider: str, model: str) -> bool:
        """Returns True if pricing data exists for this provider/model (including explicit $0.00)."""
        if (provider, model) in self.model_pricing:
            return True
        lookup_id = f"{provider}/{model}" if provider != "openrouter" else model
        if lookup_id in self.dynamic_pricing:
            return True
        normalized = self._normalize_model_version(lookup_id)
        return normalized != lookup_id and normalized in self.dynamic_pricing

    @staticmethod
    def _normalize_model_version(lookup_id: str) -> str:
        """Convert trailing version dashes to dots: anthropic/claude-haiku-4-5 -> anthropic/claude-haiku-4.5"""
        import re
        return re.sub(r'(\d)-(\d)', r'\1.\2', lookup_id)

# Expose singleton instance for the main connector logger pipeline natively
pricing_manager = PricingManager()
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
CLI scaffolding for the LLM Connector package.
Usage: llm-connector init
"""

import os
import sys
import shutil


def _get_package_conf_dir():
    """Returns the path to the frozen template configs shipped inside the pip package."""
    return os.path.join(os.path.dirname(__file__), "conf")


def _get_package_env_template():
    """Returns the path to the frozen .env.template shipped inside the pip package."""
    return os.path.join(os.path.dirname(__file__), ".env.template")


def init_project(target_dir=None):
    """Scaffold a new llm-connector workspace in the target directory."""
    if target_dir is None:
        target_dir = os.path.join(os.getcwd(), "llm-connector")

    conf_dir = os.path.join(target_dir, "conf")
    logs_dir = os.path.join(target_dir, "logs")

    if os.path.exists(target_dir):
        print(f"[!] Directory already exists: {target_dir}")
        print("    Use --force to overwrite, or delete it manually.")
        return False

    # 1. Create directory structure
    os.makedirs(conf_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    print(f"[+] Created: {target_dir}/")
    print(f"[+] Created: {conf_dir}/")
    print(f"[+] Created: {logs_dir}/")

    # 2. Copy YAML templates from the frozen package
    package_conf = _get_package_conf_dir()
    templates = ["llm.yaml", "security.yaml", "logs.yaml", "override.yaml.template"]

    for template in templates:
        src = os.path.join(package_conf, template)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(conf_dir, template))
            print(f"[+] Copied: conf/{template}")
        else:
            print(f"[!] Warning: Template not found in package: {template}")

    # 3. Copy .env.template
    env_template = _get_package_env_template()
    if os.path.exists(env_template):
        shutil.copy2(env_template, os.path.join(target_dir, ".env.template"))
        print(f"[+] Copied: .env.template")
    else:
        print(f"[!] Warning: .env.template not found in package")

    # 4. Print next steps
    print(f"\n{'='*50}")
    print("LLM Connector scaffolded successfully!")
    print(f"{'='*50}")
    print(f"\nNext steps:")
    print(f"  1. cd {os.path.basename(target_dir)}")
    print(f"  2. cp .env.template .env")
    print(f"  3. Edit .env with your API keys")
    print(f"  4. cp conf/override.yaml.template conf/override.yaml")
    print(f"  5. Edit conf/override.yaml with your local endpoints")
    print(f"\nThen use in your Python code:")
    print(f"  from connector import chat_completion")
    print(f"  response, *_ = chat_completion(messages)")

    return True


def main():
    """Entry point for the llm-connector CLI."""
    if len(sys.argv) < 2:
        print("Usage: llm-connector <command>")
        print("")
        print("Commands:")
        print("  init       Scaffold a new llm-connector workspace in the current directory")
        sys.exit(1)

    command = sys.argv[1]

    if command == "init":
        force = "--force" in sys.argv
        target = os.path.join(os.getcwd(), "llm-connector")

        if force and os.path.exists(target):
            shutil.rmtree(target)

        success = init_project(target)
        sys.exit(0 if success else 1)
    else:
        print(f"Unknown command: {command}")
        print("Available commands: init")
        sys.exit(1)


if __name__ == "__main__":
    main()
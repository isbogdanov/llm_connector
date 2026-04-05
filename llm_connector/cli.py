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
import argparse


def _get_package_conf_dir():
    """Returns the path to the frozen template configs shipped inside the pip package."""
    return os.path.join(os.path.dirname(__file__), "conf")


def _get_package_env_template():
    """Returns the path to the frozen .env.template shipped inside the pip package."""
    return os.path.join(os.path.dirname(__file__), ".env.template")


def _get_package_readme():
    """Returns the path to the README.md shipped inside the pip package."""
    return os.path.join(os.path.dirname(__file__), "README.md")


def init_project(target_dir=None, force=False):
    """Scaffold a new llm-connector workspace in the target directory."""
    if target_dir is None:
        target_dir = os.path.join(os.getcwd(), "llm-connector")

    conf_dir = os.path.join(target_dir, "conf")
    logs_dir = os.path.join(target_dir, "logs")

    if os.path.exists(target_dir):
        if force:
            shutil.rmtree(target_dir)
        else:
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

    # 4. Copy README.md
    readme = _get_package_readme()
    if os.path.exists(readme):
        shutil.copy2(readme, os.path.join(target_dir, "README.md"))
        print(f"[+] Copied: README.md")
    else:
        print(f"[!] Warning: README.md not found in package")

    # 5. Print next steps
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
    print(f"  from llm_connector import chat_completion")
    print(f"  response, *_ = chat_completion(messages)")

    return True


def main():
    """Entry point for the llm-connector CLI."""
    parser = argparse.ArgumentParser(
        prog="llm-connector",
        description="LLM Connector — a unified Python connector for multiple LLM providers.",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # init subcommand
    init_parser = subparsers.add_parser(
        "init",
        help="Scaffold a new llm-connector workspace in the current directory",
    )
    init_parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing llm-connector/ directory if it exists",
    )

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    if args.command == "init":
        target = os.path.join(os.getcwd(), "llm-connector")
        success = init_project(target, force=args.force)
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
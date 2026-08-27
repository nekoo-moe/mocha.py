#!/usr/bin/env python3
"""Turn any chat or instruction dataset into a training-ready one, and keep
the model that comes out of it working.

A single entry point over seven subcommands:

  convert    build a standardized openai_messages dataset from any source
  verify     run independent post-conversion checks on an output directory
  push       commit one or more output directories to a Hugging Face repo
  doctor     check an exported GGUF model for tool-calling problems
  template   write a Jinja chat template into a copy of a GGUF model
  login      store a Hugging Face token, or report the current one
  update     self-update the mocha.py tool to the latest version

Each subcommand carries its own options; see its --help. The modules behind
them stay importable on their own, so `from convert_dataset import
unpack_tools` keeps working for downstream code.
"""
from __future__ import annotations
import argparse
import logging
import sys
from typing import Callable, NamedTuple
import convert_dataset
import gguf_tools
import hf_auth
import push_to_hf
import verify_output

VERSION = "1.0.0"

LOG_FORMAT = '%(asctime)s | %(levelname)-7s | %(message)s'

class Command(NamedTuple):
    name: str
    summary: str
    description: str | None
    add: Callable[[argparse.ArgumentParser], None]
    run: Callable[[argparse.Namespace], int]

def add_update_arguments(parser: argparse.ArgumentParser) -> None:
    pass

def run_update(args: argparse.Namespace) -> int:
    import subprocess
    print("checking git status for self-update...", file=sys.stderr)
    try:
        subprocess.check_call(["git", "rev-parse", "--is-inside-work-tree"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        print("error: mocha.py is not running from a git repository. Cannot self-update.", file=sys.stderr)
        print("please clone the repository from GitHub: https://github.com/nekoo-moe/mocha.py.git", file=sys.stderr)
        return 1
    print("running 'git pull' to fetch and apply updates...", file=sys.stderr)
    try:
        branch = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"], text=True).strip()
        remote = subprocess.check_output(["git", "config", "--get", "remote.origin.url"], text=True).strip()
        print(f"updating from {remote} (branch: {branch})...", file=sys.stderr)
        res = subprocess.run(["git", "pull", "origin", branch], capture_output=True, text=True)
        if res.returncode == 0:
            print("update completed successfully!", file=sys.stderr)
            print(res.stdout, file=sys.stderr)
            return 0
        else:
            print("update failed. Error details:", file=sys.stderr)
            print(res.stderr, file=sys.stderr)
            return 1
    except Exception as e:
        print(f"failed to execute git pull: {e}", file=sys.stderr)
        return 1

COMMANDS = (
    Command('convert', 'build a standardized openai_messages dataset from any source', convert_dataset.__doc__, convert_dataset.add_arguments, convert_dataset.run),
    Command('verify', 'run independent post-conversion checks on an output directory', verify_output.__doc__, verify_output.add_arguments, verify_output.run),
    Command('push', 'commit one or more output directories to a Hugging Face repo', push_to_hf.__doc__, push_to_hf.add_arguments, push_to_hf.run),
    Command('doctor', 'check an exported GGUF model for tool-calling problems', gguf_tools.__doc__, gguf_tools.add_doctor_arguments, gguf_tools.run_doctor),
    Command('template', 'write a Jinja chat template into a copy of a GGUF model', gguf_tools.__doc__, gguf_tools.add_template_arguments, gguf_tools.run_template),
    Command('login', 'store a Hugging Face token, or report the current one', hf_auth.__doc__, hf_auth.add_arguments, hf_auth.run),
    Command('update', 'self-update the mocha.py tool to the latest version', 'Perform a self-update by running git pull if in a git repository.', add_update_arguments, run_update)
)
HANDLERS = {command.name: command.run for command in COMMANDS}

class Formatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawDescriptionHelpFormatter):
    pass

def print_header() -> None:
    import os
    import time
    import json
    import urllib.request
    print(f"mocha.py {VERSION} - headless cli toolbox that help you work with datasets for LLM", file=sys.stderr)
    print("Copyright (C) 2026 Alexoy Vladimirov, NekoTech Foundation & repository contributors.", file=sys.stderr)
    
    cache_dir = os.path.expanduser("~/.cache")
    cache_file = os.path.join(cache_dir, "mocha_update_check.json")
    latest_version = None
    try:
        os.makedirs(cache_dir, exist_ok=True)
        now = time.time()
        cached = {}
        if os.path.exists(cache_file):
            try:
                with open(cache_file, "r") as f:
                    cached = json.load(f)
            except Exception:
                pass
        
        if now - cached.get("last_check", 0) > 3600:
            req = urllib.request.Request(
                "https://api.github.com/repos/nekoo-moe/mocha.py/releases/latest",
                headers={"User-Agent": "mocha.py-cli-agent"}
            )
            with urllib.request.urlopen(req, timeout=1.0) as response:
                data = json.loads(response.read().decode())
                latest_version = data.get("tag_name", "").lstrip("v")
            cached = {"last_check": now, "latest_version": latest_version}
            with open(cache_file, "w") as f:
                json.dump(cached, f)
        else:
            latest_version = cached.get("latest_version")
    except Exception:
        pass
    
    if latest_version and latest_version != VERSION:
        try:
            cur_parts = [int(x) for x in VERSION.split(".")]
            new_parts = [int(x) for x in latest_version.split(".")]
            if new_parts > cur_parts:
                print(f"UPDATE: New version available ({VERSION} -> {latest_version}). Get it on official repository on Github.", file=sys.stderr)
        except Exception:
            pass

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog='mocha.py', description=__doc__, formatter_class=Formatter)
    parser.add_argument('-q', '--quiet', action='store_true', help='log warnings and errors only')
    sub = parser.add_subparsers(dest='command', metavar='COMMAND', required=True)
    for command in COMMANDS:
        child = sub.add_parser(command.name, help=command.summary, description=command.description, formatter_class=Formatter)
        command.add(child)
    return parser

def main(argv: list[str] | None=None) -> int:
    print_header()
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.WARNING if args.quiet else logging.INFO, format=LOG_FORMAT, datefmt='%H:%M:%S')
    return HANDLERS[args.command](args)

if __name__ == '__main__':
    sys.exit(main())
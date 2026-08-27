"""Hugging Face authentication, asked for only when it is actually needed.

Reading a public dataset needs no token. A private or gated one, and any
push, does. Rather than failing with a bare 401, the helpers here recognize an
authentication failure, ask for a token when a terminal is attached, store it
through huggingface_hub's own login, and let the caller retry.
"""
from __future__ import annotations
import argparse
import logging
import os
import sys
LOG = logging.getLogger('auth')
TOKEN_URL = 'https://huggingface.co/settings/tokens'
AUTH_MARKERS = ('401', '403', 'unauthorized', 'forbidden', 'gated', 'authentication', 'not authorized', 'must be authenticated', 'invalid credentials', 'repository not found')

def resolve_token(explicit: str | None=None) -> str | None:
    if explicit:
        return explicit
    for name in ('HF_TOKEN', 'HUGGING_FACE_HUB_TOKEN', 'HUGGINGFACE_TOKEN'):
        if os.environ.get(name):
            return os.environ[name]
    try:
        from huggingface_hub import get_token
        return get_token()
    except Exception:
        return None

def is_auth_error(err: BaseException) -> bool:
    from huggingface_hub.errors import GatedRepoError, RepositoryNotFoundError
    if isinstance(err, (GatedRepoError, RepositoryNotFoundError)):
        return True
    text = f'{type(err).__name__}: {err}'.lower()
    return any((marker in text for marker in AUTH_MARKERS))

def whoami(token: str | None=None) -> dict | None:
    from huggingface_hub import HfApi
    try:
        return HfApi(token=token).whoami()
    except Exception:
        return None

def ask_for_token() -> str | None:
    if not sys.stdin.isatty():
        return None
    print(f'\na Hugging Face token is required; create one at {TOKEN_URL}')
    print('it is stored by huggingface_hub, not by this tool')
    try:
        from getpass import getpass
        return getpass('token (input hidden, empty to cancel): ').strip()
    except (EOFError, KeyboardInterrupt):
        print()
        return None

def ensure_login(token: str | None=None, *, prompt: bool=True, reason: str='this operation') -> dict:
    resolved = resolve_token(token)
    who = whoami(resolved) if resolved else None
    if who is not None:
        return who
    if resolved:
        LOG.warning('the token in use was rejected')
    if prompt:
        entered = ask_for_token()
        if entered:
            from huggingface_hub import login
            login(token=entered, add_to_git_credential=False)
            who = whoami(entered)
            if who is not None:
                return who
            print('that token was rejected')
    raise SystemExit(f'{reason} needs a Hugging Face login. Run `hf auth login`, set HF_TOKEN, or pass --token. Create a token at {TOKEN_URL}')

def retry_with_login(call, *, token: str | None=None, prompt: bool=True, reason: str='this dataset'):
    resolved = resolve_token(token)
    try:
        return call(resolved)
    except Exception as err:
        if not is_auth_error(err):
            raise
        LOG.warning('%s: %s', type(err).__name__, err)
        who = ensure_login(token, prompt=prompt, reason=reason)
        LOG.info('authenticated as %s', who['name'])
        return call(resolve_token(token))

def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument('--token', default=None, metavar='TOKEN', help='token to store; prompted for when omitted')
    parser.add_argument('--status', action='store_true', help='only report who is currently logged in')
    parser.add_argument('--logout', action='store_true', help='forget the stored token')

def run(args: argparse.Namespace) -> int:
    if args.logout:
        from huggingface_hub import logout
        logout()
        print('logged out')
        return 0
    if args.status:
        who = whoami(resolve_token(args.token))
        if who is None:
            print('not logged in')
            return 1
        print(f"logged in as {who['name']}")
        for org in who.get('orgs') or []:
            print(f"  org  {org['name']}")
        return 0
    who = ensure_login(args.token, reason='login')
    print(f"logged in as {who['name']}")
    for org in who.get('orgs') or []:
        print(f"  org  {org['name']}")
    return 0
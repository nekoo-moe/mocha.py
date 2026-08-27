"""Convert a chat or instruction dataset into a standardized
`openai_messages` dataset for Unsloth / TRL ``SFTTrainer``.

Reads any Hugging Face dataset repo or local path and recognizes the common
supervised layouts:

  * ``openai``           -- a ``messages`` column of role/content dicts
  * ``openai-json``      -- the same, serialized as a JSON string
  * ``sharegpt``         -- ``conversations`` with from/value turns
  * ``alpaca``           -- ``instruction`` (+ ``input``) and ``output``
  * ``prompt-completion``-- ``prompt`` and ``completion``

Column names are detected, and every one of them can be overridden. Output is
a single uniform Arrow schema per run, written as Parquet shards in the
standard Hugging Face layout so ``load_dataset("./out")`` just works.

Parquet shards are resolved explicitly where the layout allows it, which pins
the revision and drops shard sets left behind by earlier revisions; anything
else falls back to a plain ``load_dataset``.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import logging
import math
import os
import re
import shutil
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable
LOG = logging.getLogger('convert')
LOG.addHandler(logging.NullHandler())
DEFAULT_TOKENIZER = 'Qwen/Qwen3-0.6B'
VALID_ROLES = ('system', 'user', 'assistant', 'tool')
KEEP_META = ('split', 'subset', 'domain', 'subdomain', 'category', 'topic', 'task', 'task_type', 'source', 'dataset', 'origin', 'language', 'lang', 'model', 'teacher_model', 'quality', 'score', 'sampling_weight', 'weight')
DROP_META = ('messages_json', 'tools_json', 'conversations', 'conversation', 'chat', 'instruction', 'input', 'output', 'response', 'prompt', 'completion', 'answer', 'question', 'system', 'system_prompt', 'context')
_SPLIT = '[A-Za-z0-9_]+(?:\\.[A-Za-z0-9_]+)*'
SHARD_RE = re.compile(f'^(?P<split>{_SPLIT})-(?P<idx>\\d{{5}})-of-(?P<tot>\\d{{5}})(?:-[0-9A-Za-z]+)?\\.parquet$')
PLAIN_RE = re.compile(f'^(?P<split>{_SPLIT})\\.parquet$')
FILE_BUILDERS = {'.json': 'json', '.jsonl': 'json', '.csv': 'csv', '.tsv': 'csv', '.parquet': 'parquet', '.txt': 'text', '.arrow': 'arrow'}
CONVERSATION_COLS = ('messages', 'conversations', 'conversation', 'chat', 'messages_json')
ROLE_ALIASES = {'human': 'user', 'user': 'user', 'gpt': 'assistant', 'chatgpt': 'assistant', 'assistant': 'assistant', 'bot': 'assistant', 'model': 'assistant', 'system': 'system', 'tool': 'tool', 'function': 'tool', 'observation': 'tool', 'function_call': 'assistant', 'function_response': 'tool'}
_SIZE_RE = re.compile('^\\s*(\\d+(?:\\.\\d+)?)\\s*([KMGT]?B)\\s*$', re.I)
_SIZE_MUL = {'B': 1, 'KB': 10 ** 3, 'MB': 10 ** 6, 'GB': 10 ** 9, 'TB': 10 ** 12}

def parse_size(text: str) -> int:
    m = _SIZE_RE.match(text)
    if not m:
        raise argparse.ArgumentTypeError(f"bad size {text!r} (try '500MB')")
    return int(float(m.group(1)) * _SIZE_MUL[m.group(2).upper()])

def csv_list(text: str | None) -> list[str] | None:
    if not text:
        return None
    return [x.strip() for x in text.split(',') if x.strip()]

def add_arguments(p: argparse.ArgumentParser) -> None:
    src = p.add_argument_group('source')
    src.add_argument('source', metavar='SOURCE', help='Hugging Face dataset repo id, or any local path load_dataset understands')
    src.add_argument('-c', '--config', default=None, help="dataset config to convert; default is the only one present, or the loader's default")
    src.add_argument('--revision', default='main', help='git revision of the source repo; pin it for a reproducible run')
    src.add_argument('-s', '--split', default='all', help="'all', one split name, or a comma list")
    src.add_argument('--loader', choices=('auto', 'shards', 'datasets'), default='auto', help='shards: resolve parquet files explicitly, pin the revision and drop stale shard sets; datasets: plain load_dataset; auto: shards, then fall back')
    src.add_argument('--allow-stale-shards', action='store_true', help='keep every overlapping shard set instead of the newest, reproducing raw load_dataset behaviour')
    src.add_argument('--token', default=None, metavar='TOKEN', help='Hugging Face token for a private source')
    fmt = p.add_argument_group('input format')
    fmt.add_argument('--input-format', default='auto', choices=('auto', 'openai', 'openai-json', 'sharegpt', 'alpaca', 'prompt-completion'), help='shape of the source rows')
    fmt.add_argument('--messages-column', default=None, metavar='COL', help='conversation column, for openai and sharegpt')
    fmt.add_argument('--tools-column', default=None, metavar='COL', help='tool definition column')
    fmt.add_argument('--system-column', default=None, metavar='COL', help='column holding a per-row system prompt')
    fmt.add_argument('--prompt-column', default=None, metavar='COL', help='user side of a flat pair (instruction, prompt, question)')
    fmt.add_argument('--input-column', default=None, metavar='COL', help='extra context appended to the user turn (alpaca input, context)')
    fmt.add_argument('--response-column', default=None, metavar='COL', help='assistant side of a flat pair (output, response, completion, answer)')
    fmt.add_argument('--system-prompt', default=None, metavar='TEXT', help='system turn to prepend when a row has none')
    flt = p.add_argument_group('filtering')
    flt.add_argument('-w', '--where', action='append', default=[], metavar='COL=V[,V]', help='keep rows whose COL is one of these values; repeatable')
    flt.add_argument('--where-not', action='append', default=[], metavar='COL=V[,V]', help='drop rows whose COL is one of these values; repeatable')
    flt.add_argument('--min', action='append', default=[], dest='min_value', metavar='COL=N', help='keep rows whose numeric COL is >= N; repeatable')
    flt.add_argument('--limit', type=int, default=None, metavar='N', help='keep at most N rows per split')
    shp = p.add_argument_group('shaping')
    shp.add_argument('--reasoning', default='auto', choices=('auto', 'keep', 'think-tags', 'drop'), help='auto: emit `reasoning_content` only if the source has any; think-tags: fold it into content as <think>...</think>; drop: discard it')
    shp.add_argument('--tools-format', choices=('nested', 'json-string'), default='nested', help='nested: List[Dict] with a JSON-string `parameters_json` leaf (Arrow cannot type free-form JSON Schema); json-string: the whole tool list as one string')
    shp.add_argument('--force-full-schema', action='store_true', help='always emit tool_calls/tool_call_id/name/reasoning_content even when unused, so separate runs stay concatenable')
    shp.add_argument('--keep-columns', type=csv_list, default=None, metavar='COLS', help='also carry these source columns through')
    shp.add_argument('--keep-all-metadata', action='store_true', help='carry every column that is not consumed')
    shp.add_argument('--no-tool-repair', dest='repair_tools', action='store_false', help='do not rebuild tool definitions for rows whose tools column holds invocation records or nameless placeholders')
    shp.add_argument('--strict', action='store_true', help='also drop rows with bad role ordering or dangling tool_call_ids')
    out = p.add_argument_group('output')
    out.add_argument('-o', '--output-dir', default='./converted_dataset', type=Path)
    out.add_argument('--overwrite', action='store_true')
    out.add_argument('--max-shard-size', type=parse_size, default='500MB', help='target uncompressed bytes per Parquet shard')
    out.add_argument('--push-to-hub', default=None, metavar='REPO_ID', help='also push the result straight to this dataset repo')
    out.add_argument('--hub-config-name', default='default')
    out.add_argument('--private', action='store_true')
    run = p.add_argument_group('runtime')
    run.add_argument('--num-proc', type=int, default=min(os.cpu_count() or 1, 12))
    run.add_argument('--batch-size', type=int, default=200)
    run.add_argument('--cache-dir', default=None)
    run.add_argument('--no-verify', action='store_true')
    run.add_argument('--tokenizer', default=DEFAULT_TOKENIZER, help='tokenizer used by the verification block')

@dataclass
class ShardPlan:
    files: dict[str, list[str]] = field(default_factory=dict)
    dropped_stale: dict[str, list[str]] = field(default_factory=dict)
    revision: str = 'main'
    directory: str = ''
    resolver: str = 'shards'

    @property
    def splits(self) -> list[str]:
        return list(self.files)

class LayoutError(RuntimeError):
    pass

def _index_parquet(names: Iterable[str]) -> dict[str, dict]:
    index: dict[str, dict] = {}
    for name in names:
        if not name.endswith('.parquet'):
            continue
        directory, _, base = name.rpartition('/')
        match = SHARD_RE.match(base)
        if match is not None:
            key = (match.group('split'), match.group('tot'))
            order = int(match.group('idx'))
        else:
            match = PLAIN_RE.match(base)
            if match is None:
                continue
            key, order = ((match.group('split'), '00001'), 0)
        bucket = index.setdefault(directory, {}).setdefault(key, [])
        bucket.append((order, name))
    return index

def _pick_directory(index: dict[str, dict], config: str | None, source: str) -> str:
    if not index:
        raise LayoutError(f'{source} exposes no parquet files in a layout this resolver understands')
    found = sorted(index)
    if config:
        for candidate in (f'data/{config}', config, f'{config}/data'):
            if candidate in index:
                return candidate
        tail = [d for d in found if d.rpartition('/')[2] == config]
        if len(tail) == 1:
            return tail[0]
        raise LayoutError(f'no parquet directory for config {config!r} in {source}; directories present: {found}')
    for candidate in ('data', ''):
        if candidate in index:
            return candidate
    if len(found) == 1:
        return found[0]
    raise LayoutError(f'{source} holds several parquet directories {found}; choose one with --config')

def resolve_shards(source: str, config: str | None, revision: str, wanted: list[str] | None, allow_stale: bool, token: str | None=None) -> ShardPlan:
    from huggingface_hub import HfApi
    api = HfApi(token=token)
    info = api.repo_info(source, repo_type='dataset', revision=revision)
    plan = ShardPlan(revision=info.sha or revision)
    index = _index_parquet((sib.rfilename for sib in info.siblings))
    plan.directory = _pick_directory(index, config, source)
    buckets = index[plan.directory]
    for split in sorted({split for split, _ in buckets}):
        sets = {total: items for (name, total), items in buckets.items() if name == split}
        if len(sets) > 1 and (not allow_stale):
            best = max(sets)
            stale = [f for total, items in sets.items() if total != best for _, f in sorted(items)]
            plan.dropped_stale[split] = stale
            LOG.warning('%s/%s: %d overlapping shard sets %s -> keeping -of-%s, ignoring %d stale file(s)', plan.directory or '.', split, len(sets), sorted(sets), best, len(stale))
            chosen = sets[best]
        else:
            chosen = [item for items in sets.values() for item in items]
        plan.files[split] = [f for _, f in sorted(chosen)]
    if wanted:
        missing = [s for s in wanted if s not in plan.files]
        if missing:
            raise SystemExit(f'split(s) {missing} not found; available: {plan.splits}')
        plan.files = {s: plan.files[s] for s in wanted}
    return plan

def load_source(source: str, plan: ShardPlan, cache_dir: str | None):
    from datasets import load_dataset
    data_files = {split: [f'hf://datasets/{source}@{plan.revision}/{f}' for f in files] for split, files in plan.files.items()}
    for split, files in data_files.items():
        LOG.info('split %-12s -> %d shard(s)', split, len(files))
    return load_dataset('parquet', data_files=data_files, cache_dir=cache_dir)

def load_plain(source: str, config: str | None, revision: str, wanted: list[str] | None, cache_dir: str | None, token: str | None=None):
    from datasets import Dataset, load_dataset
    kwargs: dict[str, Any] = {'cache_dir': cache_dir}
    local = Path(source)
    if local.is_file():
        builder = FILE_BUILDERS.get(local.suffix.lower())
        if builder is None:
            raise SystemExit(f'cannot load {source}: unsupported extension {local.suffix!r}. Supported: {sorted(FILE_BUILDERS)}')
        LOG.info('loading %s as %s', source, builder)
        loaded = load_dataset(builder, data_files=str(local), **kwargs)
        return {'train': loaded} if isinstance(loaded, Dataset) else dict(loaded)
    if local.exists():
        LOG.info('loading local path %s', source)
    else:
        kwargs['revision'] = revision
        if token:
            kwargs['token'] = token
    loaded = load_dataset(source, config, **kwargs)
    dsdict = {'train': loaded} if isinstance(loaded, Dataset) else dict(loaded)
    LOG.info('load_dataset returned splits %s', list(dsdict))
    if wanted:
        missing = [s for s in wanted if s not in dsdict]
        if missing:
            raise SystemExit(f'split(s) {missing} not found; available: {list(dsdict)}')
        dsdict = {s: dsdict[s] for s in wanted}
    return dsdict

def open_source(args: argparse.Namespace) -> tuple[dict, ShardPlan]:
    import hf_auth
    wanted = None if args.split.lower() == 'all' else csv_list(args.split)
    local = Path(args.source).exists()
    plan: ShardPlan | None = None
    if args.loader in ('auto', 'shards') and (not local):

        def resolve(token):
            return resolve_shards(args.source, args.config, args.revision, wanted, args.allow_stale_shards, token)
        try:
            plan = hf_auth.retry_with_login(resolve, token=args.token, reason=f'reading {args.source}')
        except LayoutError as err:
            if args.loader == 'shards':
                raise SystemExit(f'{err}') from None
            LOG.warning('%s', err)
            LOG.warning('falling back to load_dataset; stale shard sets cannot be detected on this path')
    elif args.loader == 'shards' and local:
        raise SystemExit('--loader shards works on Hub repos only')
    if plan is not None:
        return (load_source(args.source, plan, args.cache_dir), plan)

    def load(token):
        return load_plain(args.source, args.config, args.revision, wanted, args.cache_dir, token)
    dsdict = hf_auth.retry_with_login(load, token=args.token, reason=f'reading {args.source}')
    return (dsdict, ShardPlan(revision=args.revision, resolver='datasets'))

@dataclass
class SourceShape:
    variant: str
    msg_col: str | None
    tools_col: str | None
    flat_cols: dict[str, str]
    meta_cols: list[str]
    all_cols: list[str]

    @property
    def conversational(self) -> bool:
        return self.variant in ('openai', 'openai-json', 'sharegpt')

def _first(cols: list[str], *names: str) -> str | None:
    for name in names:
        if name in cols:
            return name
    return None

def _looks_serialized(features, column: str) -> bool:
    from datasets import Value
    feature = features[column]
    return isinstance(feature, Value) and 'string' in str(feature.dtype)

def _detect_variant(features, args) -> tuple[str, str | None, dict[str, str]]:
    cols = list(features)
    msg_col = args.messages_column or _first(cols, *CONVERSATION_COLS)
    prompt = args.prompt_column or _first(cols, 'prompt', 'instruction', 'question', 'query', 'input_text')
    response = args.response_column or _first(cols, 'completion', 'response', 'output', 'answer', 'chosen', 'target', 'output_text')
    chosen = args.input_format
    if chosen == 'auto':
        if msg_col:
            if _looks_serialized(features, msg_col):
                chosen = 'openai-json'
            elif msg_col in ('conversations', 'conversation'):
                chosen = 'sharegpt'
            else:
                chosen = 'openai'
        elif prompt and response:
            chosen = 'alpaca' if args.input_column or _first(cols, 'input', 'context') else 'prompt-completion'
        else:
            raise SystemExit(f'cannot tell how this dataset is shaped. Columns: {cols}. Pass --input-format with --messages-column, or --prompt-column and --response-column')
    flat: dict[str, str] = {}
    if chosen in ('alpaca', 'prompt-completion'):
        if not (prompt and response):
            raise SystemExit(f'--input-format {chosen} needs a user and an assistant column; pass --prompt-column and --response-column. Columns: {cols}')
        flat['user'] = prompt
        flat['assistant'] = response
        extra = args.input_column or _first(cols, 'input', 'context')
        if chosen == 'alpaca' and extra:
            flat['input'] = extra
        msg_col = None
    elif not msg_col:
        raise SystemExit(f'--input-format {chosen} needs a conversation column; pass --messages-column. Columns: {cols}')
    system = args.system_column or _first(cols, 'system', 'system_prompt')
    if system and system not in flat.values() and (system != msg_col):
        flat['system'] = system
    return (chosen, msg_col, flat)

def detect_shape(features, args) -> SourceShape:
    cols = list(features)
    variant, msg_col, flat = _detect_variant(features, args)
    tools_col = args.tools_column or _first(cols, 'tools', 'tools_json', 'functions')
    consumed = {msg_col, tools_col, 'id'} | set(flat.values())
    consumed.discard(None)
    extra = list(args.keep_columns or [])
    if args.keep_all_metadata:
        meta = [c for c in cols if c not in consumed and (c in extra or c not in DROP_META)]
    else:
        wanted = list(KEEP_META) + [c for c in extra if c not in KEEP_META]
        meta = [c for c in wanted if c in cols and c not in consumed]
    LOG.info('input format %s (conversation=%r tools=%r flat=%s), carrying metadata %s', variant, msg_col, tools_col, flat or None, meta)
    return SourceShape(variant, msg_col, tools_col, flat, meta, cols)

def sharegpt_turns(raw: Any) -> list[dict[str, Any]]:
    parsed = _loads(raw, 'messages')
    if not isinstance(parsed, (list, tuple)):
        raise RowError('messages_not_a_list')
    out = []
    for turn in parsed:
        if not isinstance(turn, dict):
            raise RowError('turn_not_a_dict')
        role = turn.get('from') or turn.get('role') or turn.get('speaker')
        content = turn.get('value')
        if content is None:
            content = turn.get('content')
        if content is None:
            content = turn.get('text')
        mapped = ROLE_ALIASES.get(str(role).strip().lower())
        if mapped is None:
            raise RowError('unknown_role')
        merged = {k: v for k, v in turn.items() if k in ('tool_calls', 'tool_call_id', 'name', 'reasoning_content')}
        out.append({'role': mapped, 'content': content, **merged})
    return out

def flat_turns(shape: SourceShape, batch: dict[str, list], i: int, system_prompt: str | None) -> list[dict[str, Any]]:
    cols = shape.flat_cols
    turns: list[dict[str, Any]] = []
    system = ''
    if 'system' in cols:
        system = _as_text(batch[cols['system']][i]).strip()
    if not system and system_prompt:
        system = system_prompt
    if system:
        turns.append({'role': 'system', 'content': system})
    user = _as_text(batch[cols['user']][i]).strip()
    if 'input' in cols:
        extra = _as_text(batch[cols['input']][i]).strip()
        if extra:
            user = f'{user}\n\n{extra}' if user else extra
    if not user:
        raise RowError('empty_prompt')
    turns.append({'role': 'user', 'content': user})
    answer = _as_text(batch[cols['assistant']][i])
    if not answer.strip():
        raise RowError('empty_response')
    turns.append({'role': 'assistant', 'content': answer})
    return turns

def raw_turns(shape: SourceShape, batch: dict[str, list], i: int, system_prompt: str | None) -> Any:
    if shape.variant == 'sharegpt':
        return sharegpt_turns(batch[shape.msg_col][i])
    if shape.conversational:
        turns = _loads(batch[shape.msg_col][i], 'messages')
        if system_prompt and isinstance(turns, list):
            if not (turns and isinstance(turns[0], dict) and (turns[0].get('role') == 'system')):
                turns = [{'role': 'system', 'content': system_prompt}, *turns]
        return turns
    return flat_turns(shape, batch, i, system_prompt)

class RowError(ValueError):
    pass

def _as_text(value: Any) -> str:
    if value is None:
        return ''
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)

def _loads(raw: Any, what: str) -> Any:
    if raw is None:
        raise RowError(f'null_{what}')
    if not isinstance(raw, str):
        return raw
    text = raw.strip()
    if not text:
        raise RowError(f'empty_{what}')
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError, TypeError):
        raise RowError(f'malformed_{what}_json') from None

def normalize_tool_calls(raw: Any) -> list[dict[str, Any]]:
    if not raw:
        return []
    if not isinstance(raw, (list, tuple)):
        raise RowError('bad_tool_calls')
    out = []
    for i, call in enumerate(raw):
        if not isinstance(call, dict):
            raise RowError('bad_tool_calls')
        fn = call.get('function') or {}
        if not isinstance(fn, dict) or not fn.get('name'):
            raise RowError('tool_call_without_name')
        args = fn.get('arguments')
        out.append({'id': _as_text(call.get('id')) or f'call_{i}', 'type': _as_text(call.get('type')) or 'function', 'function': {'name': _as_text(fn['name']), 'arguments': _as_text(args)}})
    return out

def _tool_def(name: str, description: str='', params_json: str='{}') -> dict:
    return {'type': 'function', 'function': {'name': name, 'description': description, 'parameters_json': params_json or '{}'}}

def normalize_tools(raw: Any, call_names: list[str], repair: bool=True) -> tuple[list[dict[str, Any]], list[str]]:
    repairs: list[str] = []
    parsed = [] if raw is None else raw if not isinstance(raw, str) else _loads(raw, 'tools')
    if parsed in ([], (), {}, '', None):
        parsed = []
    if not isinstance(parsed, (list, tuple)):
        raise RowError('bad_tools')
    defs: list[dict[str, Any]] = []
    invoked: list[str] = []
    placeholders = 0
    for tool in parsed:
        if not isinstance(tool, dict):
            raise RowError('bad_tools')
        fn = tool.get('function')
        if isinstance(fn, dict):
            name = _as_text(fn.get('name'))
            if not name:
                placeholders += 1
                continue
            if 'parameters_json' in fn and fn.get('parameters_json') is not None:
                params_json = _as_text(fn['parameters_json']) or '{}'
            else:
                params = fn.get('parameters')
                params_json = json.dumps(params if params is not None else {}, ensure_ascii=False, separators=(',', ':'))
            defs.append(_tool_def(name, _as_text(fn.get('description')), params_json))
        elif fn is None and (tool.get('name') is not None and {'arguments', 'arguments_json', 'tool_call_id'} & set(tool)):
            name = _as_text(tool.get('name'))
            if name:
                invoked.append(name)
            else:
                placeholders += 1
        else:
            raise RowError('unrecognized_tool_entry')
    if placeholders:
        repairs.append('dropped_nameless_tool_defs')
    if invoked:
        repairs.append('tools_were_invocation_records')
    if not defs and repair:
        names = list(dict.fromkeys(invoked + call_names))
        if names:
            defs = [_tool_def(n) for n in names]
            repairs.append('tools_rebuilt_from_calls')
    elif not defs and (invoked or placeholders):
        repairs.append('tools_unrecoverable')
    seen: set[str] = set()
    unique = []
    for d in defs:
        name = d['function']['name']
        if name in seen:
            repairs.append('duplicate_tool_def')
            continue
        seen.add(name)
        unique.append(d)
    missing = [n for n in dict.fromkeys(call_names) if n and n not in seen]
    if missing and repair:
        unique += [_tool_def(n) for n in missing]
        repairs.append('added_missing_called_tools')
    return (unique, sorted(set(repairs)))

def unpack_tools(tools: Any) -> list[dict[str, Any]] | None:
    if not tools:
        return None
    if isinstance(tools, str):
        tools = json.loads(tools)
        if not tools:
            return None
    out = []
    for tool in tools:
        fn = dict(tool.get('function') or {})
        blob = fn.pop('parameters_json', None)
        if blob is not None:
            fn['parameters'] = json.loads(blob) if blob else {}
        elif isinstance(fn.get('parameters'), str):
            fn['parameters'] = json.loads(fn['parameters'] or '{}')
        out.append({**tool, 'function': fn})
    return out

def normalize_messages(raw: Any, reasoning: str, want_reasoning: bool, want_tool_fields: bool, strict: bool) -> list[dict[str, Any]]:
    parsed = _loads(raw, 'messages')
    if not isinstance(parsed, (list, tuple)):
        raise RowError('messages_not_a_list')
    if len(parsed) == 0:
        raise RowError('empty_conversation')
    out: list[dict[str, Any]] = []
    seen_call_ids: set[str] = set()
    has_real_assistant = False
    for turn in parsed:
        if not isinstance(turn, dict):
            raise RowError('turn_not_a_dict')
        role = turn.get('role')
        if role not in VALID_ROLES:
            raise RowError('unknown_role')
        content = _as_text(turn.get('content'))
        reason = _as_text(turn.get('reasoning_content'))
        calls = normalize_tool_calls(turn.get('tool_calls'))
        if calls and role != 'assistant':
            raise RowError('tool_calls_on_non_assistant')
        if reason and reasoning == 'think-tags':
            content = f'<think>\n{reason}\n</think>\n\n{content}'
            reason = ''
        elif reasoning == 'drop':
            reason = ''
        msg: dict[str, Any] = {'role': role, 'content': content}
        if want_reasoning:
            msg['reasoning_content'] = reason
        if want_tool_fields:
            msg['tool_calls'] = calls
            msg['tool_call_id'] = _as_text(turn.get('tool_call_id'))
            msg['name'] = _as_text(turn.get('name'))
        elif calls:
            raise RowError('tool_calls_but_schema_has_no_tool_fields')
        if role == 'assistant':
            if content.strip() or calls or reason.strip():
                has_real_assistant = True
            seen_call_ids.update((c['id'] for c in calls))
        elif role == 'tool' and strict:
            tcid = _as_text(turn.get('tool_call_id'))
            if tcid and tcid not in seen_call_ids:
                raise RowError('dangling_tool_call_id')
        out.append(msg)
    if not has_real_assistant:
        raise RowError('no_trainable_assistant_turn')
    if strict:
        roles = [m['role'] for m in out]
        if roles[-1] != 'assistant':
            raise RowError('does_not_end_on_assistant')
        if any((a == b for a, b in zip(roles, roles[1:]) if a in ('user', 'system'))):
            raise RowError('consecutive_same_role')
    return out

def build_features(shape: SourceShape, source_features, *, want_reasoning: bool, want_tool_fields: bool, want_tools_col: bool, tools_format: str):
    from datasets import Features, List as DList, Value
    turn: dict[str, Any] = {'role': Value('string'), 'content': Value('large_string')}
    if want_reasoning:
        turn['reasoning_content'] = Value('large_string')
    if want_tool_fields:
        turn['tool_calls'] = DList({'id': Value('string'), 'type': Value('string'), 'function': {'name': Value('string'), 'arguments': Value('large_string')}})
        turn['tool_call_id'] = Value('string')
        turn['name'] = Value('string')
    feats: dict[str, Any] = {'id': Value('string'), 'messages': DList(turn)}
    if want_tools_col:
        if tools_format == 'json-string':
            feats['tools'] = Value('large_string')
        else:
            feats['tools'] = DList({'type': Value('string'), 'function': {'name': Value('string'), 'description': Value('large_string'), 'parameters_json': Value('large_string')}})
    for col in shape.meta_cols:
        feats[col] = source_features[col]
    return Features(feats)

def probe_content(dsdict, shape: SourceShape) -> tuple[bool, bool]:
    import pyarrow as pa
    import pyarrow.compute as pc
    has_reasoning = has_tools = False
    if not shape.conversational and (not shape.tools_col):
        return (False, False)
    for split, ds in dsdict.items():
        table = ds.data.table if hasattr(ds.data, 'table') else ds.data
        if shape.tools_col:
            col = table.column(shape.tools_col).combine_chunks()
            if isinstance(col, pa.ChunkedArray):
                col = col.combine_chunks()
            if pa.types.is_string(col.type) or pa.types.is_large_string(col.type):
                nonempty = pc.greater(pc.utf8_length(pc.coalesce(col, '')), 2)
            else:
                nonempty = pc.greater(pc.list_value_length(col), 0)
            if pc.any(nonempty, min_count=0).as_py():
                has_tools = True
        if not shape.conversational or shape.variant == 'sharegpt':
            continue
        col = table.column(shape.msg_col)
        if shape.variant == 'openai':
            turns = pc.list_flatten(col.combine_chunks())
            fields = [f.name for f in turns.type]
            if 'reasoning_content' in fields:
                rc = turns.field('reasoning_content')
                if pc.any(pc.greater(pc.utf8_length(pc.coalesce(rc, '')), 0), min_count=0).as_py():
                    has_reasoning = True
            if 'role' in fields:
                if pc.any(pc.equal(turns.field('role'), 'tool'), min_count=0).as_py():
                    has_tools = True
            if 'tool_calls' in fields:
                if pc.any(pc.greater(pc.list_value_length(turns.field('tool_calls')), 0), min_count=0).as_py():
                    has_tools = True
        else:
            for open_q in ('"reasoning_content":"', '"reasoning_content": "'):
                total = pc.sum(pc.count_substring(col, open_q), min_count=0).as_py() or 0
                empty = pc.sum(pc.count_substring(col, open_q + '"'), min_count=0).as_py() or 0
                if total > empty:
                    has_reasoning = True
            for needle in ('"role":"tool"', '"role": "tool"', '"tool_calls":[{', '"tool_calls": [{'):
                if pc.any(pc.match_substring(col, needle), min_count=0).as_py():
                    has_tools = True
    return (has_reasoning, has_tools)
OK_COL, REASON_COL, REPAIR_COL = ('__ok', '__drop_reason', '__repairs')

def make_transform(shape: SourceShape, *, reasoning: str, want_reasoning: bool, want_tool_fields: bool, want_tools_col: bool, tools_format: str, strict: bool, repair_tools: bool=True, system_prompt: str | None=None):
    tools_col, meta_cols = (shape.tools_col, shape.meta_cols)
    empty_tools: Any = '[]' if tools_format == 'json-string' else []

    def transform(batch: dict[str, list]) -> dict[str, list]:
        n = len(next(iter(batch.values())))
        out: dict[str, list] = {'id': [], 'messages': [], OK_COL: [], REASON_COL: [], REPAIR_COL: []}
        if want_tools_col:
            out['tools'] = []
        for col in meta_cols:
            out[col] = []
        for i in range(n):
            reason = ''
            repairs: list[str] = []
            messages: list[dict[str, Any]] = []
            tools: Any = empty_tools
            try:
                messages = normalize_messages(raw_turns(shape, batch, i, system_prompt), reasoning, want_reasoning, want_tool_fields, strict)
                if want_tools_col:
                    called = [c['function']['name'] for m in messages for c in m.get('tool_calls') or []]
                    norm, repairs = normalize_tools(batch[tools_col][i] if tools_col else None, called, repair_tools)
                    tools = json.dumps(norm, ensure_ascii=False, separators=(',', ':')) if tools_format == 'json-string' else norm
            except RowError as exc:
                reason = str(exc)
            except Exception as exc:
                reason = f'unexpected_{type(exc).__name__}'
            row_id = batch.get('id', [None] * n)[i]
            text = _as_text(row_id)
            if not text and messages:
                blob = json.dumps(messages, ensure_ascii=False, sort_keys=True).encode()
                text = hashlib.sha1(blob).hexdigest()[:32]
            out['id'].append(text)
            out['messages'].append(messages if not reason else [])
            if want_tools_col:
                out['tools'].append(tools if not reason else empty_tools)
            for col in meta_cols:
                out[col].append(batch[col][i])
            out[OK_COL].append(not reason)
            out[REASON_COL].append(reason)
            out[REPAIR_COL].append(','.join(repairs))
        return out
    return transform

def parse_predicate(spec: str, cols: set[str], kind: str):
    column, sep, raw = spec.partition('=')
    column = column.strip()
    if not sep or not column:
        raise SystemExit(f'--{kind} expects COLUMN=VALUE, got {spec!r}')
    if column not in cols:
        raise SystemExit(f'--{kind} {spec!r}: no column {column!r} in the source. Columns: {sorted(cols)}')
    if kind == 'min':
        try:
            threshold = float(raw)
        except ValueError:
            raise SystemExit(f'--min {spec!r}: {raw!r} is not a number') from None
        return (column, lambda v: v is not None and float(v) >= threshold)
    values = {v.strip() for v in raw.split(',') if v.strip()}
    if not values:
        raise SystemExit(f'--{kind} {spec!r} lists no values')
    if kind == 'where':
        return (column, lambda v: _as_text(v) in values)
    return (column, lambda v: _as_text(v) not in values)

def apply_filters(ds, args, cols: Iterable[str]):
    cols = set(cols)
    preds = [parse_predicate(spec, cols, 'where') for spec in args.where]
    preds += [parse_predicate(spec, cols, 'where-not') for spec in args.where_not]
    preds += [parse_predicate(spec, cols, 'min') for spec in args.min_value]
    if not preds:
        return ds

    def keep_batch(batch):
        n = len(next(iter(batch.values())))
        return [all((fn(batch[col][i]) for col, fn in preds)) for i in range(n)]
    return ds.filter(keep_batch, batched=True, batch_size=1000, desc='filtering metadata')

def fidelity_check(src_ds, out_ds, shape: SourceShape, sample: int=24) -> int:
    n = min(len(src_ds), len(out_ds))
    if n == 0:
        return 0
    step = max(1, n // sample)
    checked = 0
    for i in list(range(0, n, step))[:sample]:
        got = out_ds[i]['messages']
        if not got:
            continue
        if not shape.conversational:
            row = src_ds[i]
            prompt = _as_text(row[shape.flat_cols['user']]).strip()
            answer = _as_text(row[shape.flat_cols['assistant']]).strip()
            joined = '\n'.join((turn['content'] for turn in got))
            assert prompt in joined, f'row {i}: prompt not preserved'
            assert answer in joined, f'row {i}: response not preserved'
            checked += 1
            continue
        raw = src_ds[i][shape.msg_col]
        try:
            src_msgs = raw if not isinstance(raw, str) else json.loads(raw)
        except Exception:
            continue
        if shape.variant == 'sharegpt':
            texts = [_as_text(t.get('value') if isinstance(t, dict) else t) for t in src_msgs]
            joined = '\n'.join((turn['content'] for turn in got))
            for text in texts:
                assert text in joined, f'row {i}: turn text not preserved'
            checked += 1
            continue
        offset = len(got) - len(src_msgs)
        assert offset in (0, 1), f'row {i}: turn count changed'
        for a, b in zip(src_msgs, got[offset:]):
            assert a.get('role') == b['role'], f'row {i}: role changed'
            src_text = _as_text(a.get('content'))
            if b['content'] != src_text:
                assert _as_text(a.get('reasoning_content')) in b['content'], f'row {i}: content not preserved'
                assert src_text in b['content'], f'row {i}: content truncated'
        checked += 1
    return checked

def write_parquet(ds, split: str, out_dir: Path, max_shard_bytes: int) -> list[Path]:
    data_dir = out_dir / 'data'
    data_dir.mkdir(parents=True, exist_ok=True)
    nbytes = ds.data.nbytes or 1
    n_shards = max(1, math.ceil(nbytes / max_shard_bytes))
    paths = []
    for i in range(n_shards):
        shard = ds.shard(num_shards=n_shards, index=i, contiguous=True)
        path = data_dir / f'{split}-{i:05d}-of-{n_shards:05d}.parquet'
        shard.to_parquet(str(path))
        paths.append(path)
    LOG.info('wrote %-10s %7d rows -> %d shard(s), %.1f MB on disk', split, len(ds), n_shards, sum((p.stat().st_size for p in paths)) / 1000000.0)
    return paths

def write_readme(out_dir: Path, splits: list[str], report: dict) -> None:
    files = {s: sorted((p.name for p in (out_dir / 'data').glob(f'{s}-*.parquet'))) for s in splits}
    lines = ['---', 'configs:', '  - config_name: default', '    data_files:']
    for split in splits:
        lines.append(f'      - split: {split}')
        lines.append('        path:')
        lines += [f'          - data/{name}' for name in files[split]]
    lines += ['---', '', f"# {report['source']['repo_id']} -> openai_messages", '', f"Converted with `mocha.py convert` (input format `{report['source'].get('input_format', '?')}`, source config `{report['source']['config']}`, revision `{report['source']['revision'][:12]}`).", '', '```python', 'from datasets import load_dataset', f'ds = load_dataset("{out_dir.as_posix()}", split="{splits[0]}")', '```', '']
    if report['schema'].get('tools_column') == 'nested':
        lines += ['`tools[*].function.parameters_json` holds the JSON-Schema blob as a', 'string (Arrow has no variant type). Restore it with `unpack_tools()`', 'from `convert_dataset` before calling `apply_chat_template(..., tools=...)`.', '']
    lines += ['## Row counts', '', '| split | rows |', '|---|---:|']
    for split in splits:
        lines.append(f"| {split} | {report['splits'][split]['kept']:,} |")
    (out_dir / 'README.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')

def verify(out_dir: Path, tokenizer_name: str, split: str) -> bool:
    from datasets import load_dataset
    LOG.info('=' * 72)
    LOG.info('VERIFY  %s  (split=%s, tokenizer=%s)', out_dir, split, tokenizer_name)
    LOG.info('=' * 72)
    ds = load_dataset(str(out_dir), split=split)
    LOG.info('reloaded %d rows', len(ds))
    LOG.info('features:\n%s', json.dumps({k: str(v) for k, v in ds.features.items()}, indent=2))
    assert 'messages' in ds.features, 'missing `messages` column'
    row = ds[0]
    msgs = row['messages']
    assert isinstance(msgs, list) and isinstance(msgs[0], dict), 'messages not List[Dict]'
    assert msgs[0]['role'] in VALID_ROLES, f"bad role {msgs[0]['role']!r}"
    for turn in msgs:
        assert isinstance(turn.get('content'), str), 'content is not a str'
    plain_row, tool_row = (None, None)
    probe = ds.select(range(min(len(ds), 5000)))
    tools_col = probe['tools'] if 'tools' in ds.features else [None] * len(probe)
    for i, tools in enumerate(tools_col):
        if tools and tools != '[]':
            tool_row = tool_row or probe[i]
        else:
            plain_row = plain_row or probe[i]
        if plain_row is not None and tool_row is not None:
            break
    plain_row = plain_row or row
    LOG.info('samples: plain id=%s | tools id=%s', plain_row.get('id'), tool_row['id'] if tool_row else 'none found')
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=False)
    except Exception as exc:
        LOG.error('could not load tokenizer %s: %s', tokenizer_name, exc)
        return False
    ok = True
    for label, sample in (('plain', plain_row), ('tools', tool_row)):
        if sample is None:
            continue
        kwargs: dict[str, Any] = {}
        if label == 'tools':
            kwargs['tools'] = unpack_tools(sample['tools'])
            assert all((t['function']['name'] for t in kwargs['tools'])), 'tool definition with an empty name would render a broken prompt'
            assert all((isinstance(t['function']['parameters'], dict) for t in kwargs['tools'])), 'parameters did not unpack to dict'
        text = tok.apply_chat_template(sample['messages'], tokenize=False, add_generation_prompt=False, **kwargs)
        enc = tok.apply_chat_template(sample['messages'], tokenize=True, return_dict=True, **kwargs)
        ids = enc['input_ids']
        if ids and isinstance(ids[0], (list, tuple)):
            ids = ids[0]
        assert isinstance(text, str) and text, f'{label}: empty render'
        assert len(ids) > 0, f'{label}: empty tokenization'
        last = [m for m in sample['messages'] if m['role'] == 'assistant'][-1]
        probe_txt = (last['content'] or '')[:80].strip()
        if probe_txt and probe_txt not in text:
            LOG.error('%s: assistant content missing from render', label)
            ok = False
        if label == 'tools' and kwargs['tools']:
            name = kwargs['tools'][0]['function']['name']
            if name not in text:
                LOG.error('tools: tool name %r absent from render', name)
                ok = False
        reasoned = [m for m in sample['messages'] if (m.get('reasoning_content') or '').strip()]
        if reasoned and reasoned[-1]['reasoning_content'][:60].strip() not in text:
            LOG.warning('%s: `reasoning_content` is present but this template does not render it (traces would be excluded from the loss)', label)
        LOG.info('%-5s render OK: %d chars, %d tokens, %d turns', label, len(text), len(ids), len(sample['messages']))
        LOG.info('--- %s render (first 420 chars) ---\n%s\n...', label, text[:420])
    LOG.info('verification %s', 'PASSED' if ok else 'FAILED')
    return ok

def run(args: argparse.Namespace) -> int:
    t0 = time.time()
    dsdict, plan = open_source(args)
    splits = list(dsdict)
    LOG.info('source %s@%s config=%s splits=%s (%s)', args.source, plan.revision[:12], args.config, splits, plan.resolver)
    first = dsdict[splits[0]]
    shape = detect_shape(first.features, args)
    has_reasoning, has_tools = probe_content(dsdict, shape)
    want_reasoning = args.force_full_schema or (has_reasoning and args.reasoning in ('auto', 'keep'))
    want_tool_fields = args.force_full_schema or has_tools
    want_tools_col = args.force_full_schema or has_tools
    if want_reasoning:
        LOG.warning('this config keeps chain-of-thought in a separate `reasoning_content` field; most chat templates (Qwen, Llama, Mistral) ignore it, so the traces would not reach the loss. Pass --reasoning think-tags to inline them as <think>...</think>, or --reasoning drop to train on finals only.')
    LOG.info('probe: reasoning_content=%s tool_content=%s -> emit reasoning=%s tool_fields=%s tools_col=%s (%s)', has_reasoning, has_tools, want_reasoning, want_tool_fields, want_tools_col, args.tools_format)
    features = build_features(shape, first.features, want_reasoning=want_reasoning, want_tool_fields=want_tool_fields, want_tools_col=want_tools_col, tools_format=args.tools_format)
    from datasets import DatasetDict, Features, Value
    map_features = Features({**features, OK_COL: Value('bool'), REASON_COL: Value('string'), REPAIR_COL: Value('string')})
    transform = make_transform(shape, reasoning=args.reasoning, want_reasoning=want_reasoning, want_tool_fields=want_tool_fields, want_tools_col=want_tools_col, tools_format=args.tools_format, strict=args.strict, repair_tools=args.repair_tools, system_prompt=args.system_prompt)
    out_dir: Path = args.output_dir
    if out_dir.exists():
        if not args.overwrite and any(out_dir.iterdir()):
            raise SystemExit(f'{out_dir} is not empty (pass --overwrite)')
        if args.overwrite:
            shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {'source': {'repo_id': args.source, 'config': args.config, 'revision': plan.revision, 'variant': shape.variant, 'resolver': plan.resolver, 'directory': plan.directory, 'input_format': shape.variant, 'columns': {'messages': shape.msg_col, 'tools': shape.tools_col, 'flat': shape.flat_cols, 'metadata': shape.meta_cols}, 'stale_shards_ignored': plan.dropped_stale}, 'options': {k: v if isinstance(v, (str, int, float, bool, type(None), list, dict)) else str(v) for k, v in vars(args).items()}, 'schema': {'tools_column': args.tools_format if want_tools_col else None, 'reasoning_content': want_reasoning, 'tool_fields': want_tool_fields, 'arrow': str(features.arrow_schema)}, 'splits': {}, 'totals': {}, 'drop_reasons': {}, 'repairs': {}}
    all_reasons: Counter = Counter()
    all_repairs: Counter = Counter()
    converted: dict[str, Any] = {}
    for split in splits:
        ds = dsdict[split]
        n_source = len(ds)
        ds = apply_filters(ds, args, shape.all_cols)
        n_filtered = len(ds)
        if args.limit is not None:
            ds = ds.select(range(min(args.limit, len(ds))))
        n_in = len(ds)
        num_proc = max(1, min(args.num_proc, max(1, n_in // 64)))
        mapped = ds.map(transform, batched=True, batch_size=args.batch_size, num_proc=num_proc if num_proc > 1 else None, remove_columns=ds.column_names, features=map_features, desc=f'converting {split}')
        reasons = Counter((r for r in mapped[REASON_COL] if r))
        all_reasons.update(reasons)
        repairs = Counter((tag for cell in mapped[REPAIR_COL] if cell for tag in cell.split(',')))
        all_repairs.update(repairs)
        checked = fidelity_check(ds, mapped, shape)
        kept = mapped.filter(lambda b: b[OK_COL], batched=True, batch_size=1000, desc=f'dropping bad rows in {split}')
        kept = kept.remove_columns([OK_COL, REASON_COL, REPAIR_COL])
        assert kept.features == features, 'output schema drifted'
        converted[split] = kept
        report['splits'][split] = {'source_rows': n_source, 'after_metadata_filters': n_filtered, 'converted_in': n_in, 'kept': len(kept), 'dropped': n_in - len(kept), 'fidelity_samples_checked': checked, 'drop_reasons': dict(reasons), 'repairs': dict(repairs)}
        LOG.info('%-10s source=%d filtered=%d kept=%d dropped=%d (fidelity ok on %d samples)', split, n_source, n_filtered, len(kept), n_in - len(kept), checked)
        for reason, count in reasons.most_common():
            LOG.warning('  dropped  %6d rows: %s', count, reason)
        for tag, count in repairs.most_common():
            LOG.info('  repaired %6d rows: %s', count, tag)
    if all_reasons.get('tool_calls_but_schema_has_no_tool_fields'):
        raise SystemExit('tool content found after the probe said there was none; re-run with --force-full-schema')
    for split, ds in converted.items():
        write_parquet(ds, split, out_dir, args.max_shard_size)
    report['totals'] = {'kept': sum((v['kept'] for v in report['splits'].values())), 'dropped': sum((v['dropped'] for v in report['splits'].values()))}
    report['drop_reasons'] = dict(all_reasons)
    report['repairs'] = dict(all_repairs)
    report['elapsed_seconds'] = round(time.time() - t0, 1)
    write_readme(out_dir, list(converted), report)
    (out_dir / 'conversion_report.json').write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    LOG.info('kept %d rows total, dropped %d, in %.1fs', report['totals']['kept'], report['totals']['dropped'], report['elapsed_seconds'])
    if args.push_to_hub:
        import hf_auth
        who = hf_auth.ensure_login(args.token, reason=f'pushing to {args.push_to_hub}')
        LOG.info('pushing to %s as %s (private=%s)', args.push_to_hub, who['name'], args.private)
        DatasetDict(converted).push_to_hub(args.push_to_hub, config_name=args.hub_config_name, private=args.private, max_shard_size=args.max_shard_size, token=hf_auth.resolve_token(args.token))
        LOG.info('pushed https://huggingface.co/datasets/%s', args.push_to_hub)
    if not args.no_verify:
        if not verify(out_dir, args.tokenizer, list(converted)[0]):
            return 1
    return 0
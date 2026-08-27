"""Commit converted output directories to a Hugging Face dataset repo.

Every invocation is one commit on one branch. Unchanged files are skipped by
the Hub, so re-running after rebuilding a single split uploads only what moved.

Files under the directories being published that exist in the repo but no
longer exist locally are deleted in the same commit. This matters: a leftover
`data/train-00000-of-00005.parquet` sitting next to a freshly written
`data/train-00000-of-00001.parquet` is matched by the same config glob and
silently doubles the row count. Nothing outside those directories is touched
unless --prune-all is passed.

The dataset card is generated from each directory's conversion_report.json and
from the shard files actually on disk, so config names, shard paths and row
counts cannot drift from what was written.
"""
from __future__ import annotations
import argparse
import json
import re
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
SKIP_DIRS = {'.git', '.venv', 'venv', '__pycache__', '.ipynb_checkpoints', '.mypy_cache', '.ruff_cache', '.pytest_cache', '.cache'}
SKIP_SUFFIXES = ('.pyc', '.pyo', '.swp', '.tmp', '.lock')
SOURCE_FILES = ('mocha.py', 'convert_dataset.py', 'verify_output.py', 'push_to_hf.py', 'gguf_tools.py', 'hf_auth.py', 'tool_template.jinja')
GITIGNORE = '.venv/\n__pycache__/\n*.pyc\n'
SHARD_RE = re.compile('^(?P<split>.+?)-\\d{5}-of-\\d{5}\\.parquet$')
SPLIT_ORDER = {'train': 0, 'validation': 1, 'valid': 2, 'dev': 3, 'test': 4}
SAFE_NAME = re.compile('[^0-9a-zA-Z._-]+')

@dataclass
class Output:
    path: Path
    config: str
    report: dict
    shards: dict[str, list[str]] = field(default_factory=dict)

    @property
    def prefix(self) -> str:
        return self.path.name

    @property
    def rows(self) -> int:
        return int(self.report.get('totals', {}).get('kept', 0))

def find_shards(data: Path) -> dict[str, list[str]]:
    if not data.is_dir():
        sys.exit(f'missing {data} -- run the convert command first')
    found: dict[str, list[str]] = {}
    for path in sorted(data.glob('*.parquet')):
        match = SHARD_RE.match(path.name)
        if match is None:
            sys.exit(f'unexpected shard name {path}; expected <split>-NNNNN-of-NNNNN.parquet')
        found.setdefault(match.group('split'), []).append(path.name)
    if not found:
        sys.exit(f'no parquet shards in {data} -- run convert first')
    return dict(sorted(found.items(), key=lambda kv: (SPLIT_ORDER.get(kv[0], 9), kv[0])))

def config_name(path: Path, report: dict, only: bool) -> str:
    if only:
        return 'default'
    source = report.get('source', {}) or {}
    for candidate in (source.get('config'), path.name):
        if candidate:
            name = SAFE_NAME.sub('_', str(candidate)).strip('_')
            if name:
                return name
    return 'default'

def read_outputs(specs: list[str]) -> list[Output]:
    parsed: list[tuple[Path, str | None]] = []
    for spec in specs:
        raw, sep, name = spec.partition('=')
        parsed.append((Path(raw), name if sep and name else None))
    outputs: list[Output] = []
    for path, explicit in parsed:
        report_path = path / 'conversion_report.json'
        if not report_path.is_file():
            sys.exit(f'{path} has no conversion_report.json; it does not look like a converted output directory')
        report = json.loads(report_path.read_text())
        shards = find_shards(path / 'data')
        declared = set(report.get('splits') or {})
        if declared and declared != set(shards):
            sys.exit(f'{path}: the report lists splits {sorted(declared)} but data/ holds {sorted(shards)} -- re-run the conversion')
        name = explicit or config_name(path, report, only=len(parsed) == 1)
        outputs.append(Output(path=path, config=name, report=report, shards=shards))
    seen: dict[str, Path] = {}
    for out in outputs:
        if out.config in seen:
            sys.exit(f'{out.path} and {seen[out.config]} would both be published as config {out.config!r}; name one with DIR=CONFIG')
        seen[out.config] = out.path
    names = [out.prefix for out in outputs]
    if len(set(names)) != len(names):
        sys.exit(f'two output directories share a name {names}; rename one, since the name is also the path inside the repo')
    return outputs

def configs_yaml(outputs: list[Output]) -> str:
    lines = ['configs:']
    for i, out in enumerate(outputs):
        lines.append(f'  - config_name: {out.config}')
        if i == 0:
            lines.append('    default: true')
        lines.append('    data_files:')
        for split, names in out.shards.items():
            lines.append(f'      - split: {split}')
            lines.append('        path:')
            for name in names:
                lines.append(f'          - {out.prefix}/data/{name}')
    return '\n'.join(lines)

def size_category(rows: int) -> str:
    for limit, label in ((1000, 'n<1K'), (10000, '1K<n<10K'), (100000, '10K<n<100K'), (1000000, '100K<n<1M'), (10000000, '1M<n<10M')):
        if rows < limit:
            return label
    return 'n>10M'

def front_matter(outputs: list[Output], args: argparse.Namespace) -> str:
    total = sum((out.rows for out in outputs))
    lines = ['---', f'license: {args.license}']
    if args.language:
        lines.append('language:')
        lines += [f'  - {code}' for code in args.language]
    lines += ['task_categories:', '  - text-generation']
    tags = list(dict.fromkeys(['sft', 'openai-messages', *(args.tag or [])]))
    if any((out.report.get('schema', {}).get('tools_column') for out in outputs)):
        tags.append('tool-use')
    lines.append('tags:')
    lines += [f'  - {tag}' for tag in tags]
    lines += ['size_categories:', f'  - {size_category(total)}', configs_yaml(outputs), '---']
    return '\n'.join(lines)

def source_row(out: Output) -> str:
    source = out.report.get('source', {}) or {}
    repo = source.get('repo_id', '?')
    config = source.get('config') or 'default'
    revision = str(source.get('revision') or '')[:12]
    splits = ', '.join((f"{split} ({out.report['splits'][split]['kept']:,})" for split in out.shards if split in (out.report.get('splits') or {})))
    return f"| `{out.config}` | {splits or ', '.join(out.shards)} | {out.rows:,} | `{repo}` | `{config}` | `{revision}` |"

def body(outputs: list[Output], repo_id: str, args: argparse.Namespace) -> str:
    first = outputs[0]
    split = next(iter(first.shards))
    has_tools = any((out.report.get('schema', {}).get('tools_column') == 'nested' for out in outputs))
    dropped = sum((int(out.report.get('totals', {}).get('dropped', 0)) for out in outputs))
    stale = sum((len(files) for out in outputs for files in (out.report.get('source', {}).get('stale_shards_ignored') or {}).values()))
    lines = [f'# {repo_id}', '', 'Standardized to the `openai_messages` format: one `messages` column of role/content turns, ready for Unsloth and `trl.SFTTrainer`.', '', '## Configs', '', '| config | splits (rows) | total rows | source | source config | revision |', '|---|---|---:|---|---|---|']
    lines += [source_row(out) for out in outputs]
    lines += ['', '```python', 'from datasets import load_dataset', '', f'ds = load_dataset("{repo_id}", "{first.config}", split="{split}")', 'ds[0]["messages"]   # [{"role": "user", "content": ...}, ...]', '```', '', '## Schema', '', '- `messages` -- native `List[Dict]` with `role` and `content`, plus `tool_calls`, `tool_call_id` and `name` where the source had them. Text is preserved byte for byte.', "- Per-row metadata is carried through where the source had it; each config's `conversion_report.json` records exactly which columns were kept."]
    if has_tools:
        lines += ['- `tools` -- native `List[Dict]` in OpenAI function-calling shape. The JSON-Schema leaf is kept as a **string** in `tools[*].function.parameters_json`, because free-form schemas infer a different Arrow struct per shard and would make the parquet files unmergeable. Restore it before rendering:', '', '```python', 'from convert_dataset import unpack_tools', '', 'tools = unpack_tools(row["tools"])', 'tokenizer.apply_chat_template(row["messages"], tools=tools, tokenize=False)', '```']
    lines += ['', '## How this was built', '', 'Each config lists its shards by filename rather than globbing `data/`, so a file left behind by an earlier build cannot be pulled into a split.']
    if stale:
        lines.append(f'{stale} stale shard file(s) in the source were detected and excluded; see `stale_shards_ignored` in the reports.')
    lines.append('No rows were dropped in conversion.' if not dropped else f'{dropped:,} row(s) were dropped in conversion; the reports carry the reasons.')
    lines += ['', 'Every directory here carries a `conversion_report.json` with the source revision, the options used, per-split counts, drop reasons and repairs.', '', 'Produced with [mocha.py](https://github.com/nekoo-moe/mocha.py): `mocha.py convert`, checked with `mocha.py verify`, published with `mocha.py push`.']
    if args.card_body:
        lines += ['', Path(args.card_body).read_text().strip()]
    return '\n'.join(lines) + '\n'

def card(outputs: list[Output], repo_id: str, args: argparse.Namespace) -> str:
    return front_matter(outputs, args) + '\n' + body(outputs, repo_id, args)

def human(size: float) -> str:
    for unit in ('B', 'KB', 'MB', 'GB'):
        if size < 1024 or unit == 'GB':
            return f'{size:.0f}{unit}' if unit == 'B' else f'{size:.1f}{unit}'
        size /= 1024.0
    return f'{size:.1f}GB'

def walk(root: Path, prefix: str) -> list[tuple[str, Path]]:
    found: list[tuple[str, Path]] = []
    for path in sorted(root.rglob('*')):
        rel = path.relative_to(root)
        if any((part in SKIP_DIRS for part in rel.parts)):
            continue
        if not path.is_file() or path.is_symlink():
            continue
        if path.name.endswith(SKIP_SUFFIXES) or path.name == '.DS_Store':
            continue
        found.append((f'{prefix}/{rel.as_posix()}' if prefix else rel.as_posix(), path))
    return found

def collect(outputs: list[Output], args: argparse.Namespace, staged: Path) -> list[tuple[str, Path]]:
    uploads: list[tuple[str, Path]] = []
    for out in outputs:
        uploads += walk(out.path, out.prefix)
    if args.include_source:
        here = Path(__file__).parent.resolve()
        for name in SOURCE_FILES:
            path = here / name
            if path.is_file():
                uploads.append((name, path))
    for spec in args.include or []:
        path = Path(spec)
        if not path.exists():
            sys.exit(f'--include {spec}: no such file or directory')
        if path.is_dir():
            uploads += walk(path, path.name)
        else:
            uploads.append((path.name, path))
    for name in ('README.md', '.gitignore'):
        if (staged / name).is_file():
            uploads.append((name, staged / name))
    seen: dict[str, Path] = {}
    for rel, path in uploads:
        if rel in seen and seen[rel] != path:
            sys.exit(f'two files would be uploaded to {rel}: {seen[rel]} and {path}')
        seen[rel] = path
    return sorted(seen.items())

def obsolete_files(remote: set[str], uploads: list[tuple[str, Path]], outputs: list[Output], prune_all: bool) -> list[str]:
    keep = {rel for rel, _ in uploads} | {'.gitattributes'}
    if prune_all:
        return sorted((f for f in remote if f not in keep))
    managed = tuple((f'{out.prefix}/' for out in outputs))
    return sorted((f for f in remote if f.startswith(managed) and f not in keep))

def show_plan(uploads: list[tuple[str, Path]], remote: set[str], obsolete: list[str]) -> None:
    total = sum((path.stat().st_size for _, path in uploads))
    print(f'\n{len(uploads)} file(s) staged ({human(total)}), {len(obsolete)} to delete:')
    for rel, path in uploads:
        state = 'sync' if rel in remote else ' new'
        print(f'  {state}  {rel}  ({human(path.stat().st_size)})')
    for rel in obsolete:
        print(f'   del  {rel}')
    print('\nunchanged files are skipped by the Hub at commit time')

def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument('dirs', nargs='*', default=['./converted_dataset'], metavar='DIR[=CONFIG]', help="converted output directories to publish; the config name defaults to the source config, or 'default' for a single directory")
    target = parser.add_argument_group('target')
    target.add_argument('-r', '--repo', required=True, metavar='REPO_ID', help='dataset repo to commit to, e.g. me/my-dataset')
    target.add_argument('-m', '--message', required=True, metavar='MSG', help='commit message')
    target.add_argument('-b', '--branch', default='main', metavar='REF', help='branch to commit on')
    target.add_argument('--create', action='store_true', help='create the repo first if it does not exist')
    target.add_argument('--public', action='store_true', help='with --create: create it public, not private')
    target.add_argument('--token', default=None, metavar='TOKEN', help='Hugging Face token; default is the cached login or HF_TOKEN')
    card_group = parser.add_argument_group('dataset card')
    card_group.add_argument('--license', default='other', metavar='ID', help='license field of the card front matter')
    card_group.add_argument('--language', action='append', default=None, metavar='CODE', help='language code for the front matter; repeatable')
    card_group.add_argument('--tag', action='append', default=None, metavar='TAG', help='extra card tag; repeatable')
    card_group.add_argument('--card-body', default=None, metavar='FILE', help='markdown appended to the generated card')
    card_group.add_argument('--no-card', action='store_true', help="leave the repo's README.md alone")
    card_group.add_argument('--print-card', action='store_true', help='print the generated card and stop')
    step = parser.add_argument_group('behaviour')
    step.add_argument('--include', action='append', default=None, metavar='PATH', help='extra file or directory to upload; repeatable')
    step.add_argument('--include-source', action='store_true', help='also upload the mocha.py modules, so consumers can import unpack_tools from the repo')
    step.add_argument('--prune-all', action='store_true', help='also delete repo files outside the published directories')
    step.add_argument('--no-prune', action='store_true', help='delete nothing')
    step.add_argument('-n', '--dry-run', action='store_true', help='print the commit plan and stop')

def run(args: argparse.Namespace) -> int:
    from huggingface_hub import CommitOperationAdd, CommitOperationDelete, HfApi
    import hf_auth
    outputs = read_outputs(list(args.dirs))
    text = card(outputs, args.repo, args)
    if args.print_card:
        print(text)
        return 0
    who = hf_auth.ensure_login(args.token, reason=f'pushing to {args.repo}')
    token = hf_auth.resolve_token(args.token)
    api = HfApi(token=token)
    print(f"authenticated as {who['name']}")
    exists = api.repo_exists(args.repo, repo_type='dataset')
    if args.create and (not exists):
        if args.dry_run:
            print(f"would create {args.repo} ({('public' if args.public else 'private')})")
        else:
            url = api.create_repo(args.repo, repo_type='dataset', private=not args.public, exist_ok=True)
            print(f'created {url}')
    elif not exists:
        sys.exit(f'{args.repo} does not exist, or this token cannot see it -- pass --create to create it')
    else:
        if args.create:
            print(f'{args.repo} already exists -- committing to it')
        if args.public:
            print('note: --public only applies at creation; visibility left unchanged')
    remote: set[str] = set()
    if exists:
        try:
            remote = set(api.list_repo_files(args.repo, repo_type='dataset', revision=args.branch))
        except Exception as err:
            print(f'cannot list {args.repo}@{args.branch}: {err}')
            if not args.dry_run:
                api.create_branch(args.repo, branch=args.branch, repo_type='dataset', exist_ok=True)
                print(f'created branch {args.branch}')
    for out in outputs:
        print(f'  {out.prefix}/ -> config {out.config} ({out.rows:,} rows, {len(out.shards)} split(s))')
    with tempfile.TemporaryDirectory() as tmp:
        staged = Path(tmp)
        if not args.no_card:
            (staged / 'README.md').write_text(text)
        if '.gitignore' not in remote:
            (staged / '.gitignore').write_text(GITIGNORE)
        uploads = collect(outputs, args, staged)
        obsolete = [] if args.no_prune else obsolete_files(remote, uploads, outputs, args.prune_all)
        show_plan(uploads, remote, obsolete)
        if args.dry_run:
            print('\ndry run -- nothing was pushed')
            return 0
        operations = [CommitOperationAdd(path_in_repo=rel, path_or_fileobj=str(path)) for rel, path in uploads]
        operations += [CommitOperationDelete(path_in_repo=rel) for rel in obsolete]
        print(f'\ncommitting to {args.repo}@{args.branch} ...')
        info = api.create_commit(repo_id=args.repo, repo_type='dataset', revision=args.branch, operations=operations, commit_message=args.message)
    print(f'commit:  {info.commit_url}')
    print(f'dataset: https://huggingface.co/datasets/{args.repo}/tree/{args.branch}')
    return 0
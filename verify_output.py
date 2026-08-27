"""Independent post-conversion checks on a converted output directory.

Deliberately does not import the converter's state: everything is re-derived
from the parquet shards on disk and from conversion_report.json, so a bug in
the converter cannot hide itself here.

Five checks run in order: identical Arrow schema across shards,
CastError-free concatenation, the OpenAI message contract, byte-level
fidelity against the pinned source shards, and apply_chat_template across
tokenizers. The last two reach the network; skip them for an offline run.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from convert_dataset import load_plain, load_source, resolve_shards, unpack_tools
TOKENIZERS = ['Qwen/Qwen3-0.6B', 'NousResearch/Meta-Llama-3.1-8B-Instruct', 'NousResearch/Hermes-3-Llama-3.1-8B']

class Report:

    def __init__(self) -> None:
        self.fails: list[str] = []

    def check(self, cond: bool, msg: str, otherwise: str | None=None) -> None:
        failed = otherwise if otherwise is not None else msg
        print('  PASS  ' + msg if cond else '  FAIL  ' + failed)
        if not cond:
            self.fails.append(failed)

    @staticmethod
    def note(msg: str) -> None:
        print(f'  note   {msg}')

    @staticmethod
    def skip(msg: str) -> None:
        print(f'  SKIP   {msg}')

def check_shard_schemas(out_dir: Path, rep: Report, args: argparse.Namespace) -> None:
    import pyarrow.parquet as pq
    shards = sorted((out_dir / 'data').glob('*.parquet'))
    if not shards:
        rep.check(False, f"no parquet shards under {out_dir / 'data'}")
        return
    ref_name, ref = (shards[0].name, pq.ParquetFile(shards[0]).schema_arrow)
    for path in shards[1:]:
        other = pq.ParquetFile(path).schema_arrow
        rep.check(ref.equals(other), f'{path.name} matches {ref_name}')
    rep.check(len(shards) >= 1, f'{len(shards)} shard(s) present')

def check_concat(out_dir: Path, rep: Report, args: argparse.Namespace) -> None:
    from datasets import concatenate_datasets, load_dataset
    dsd = load_dataset(str(out_dir))
    try:
        merged = concatenate_datasets(list(dsd.values()))
        rep.check(len(merged) == sum((len(d) for d in dsd.values())), f'concatenated {len(dsd)} split(s) -> {len(merged)} rows')
    except Exception as exc:
        rep.check(False, f'concatenate failed: {type(exc).__name__}: {exc}')

def check_structure(out_dir: Path, rep: Report, args: argparse.Namespace) -> None:
    from datasets import load_dataset
    ds = load_dataset(str(out_dir), split=list(load_dataset(str(out_dir)))[0])
    sample = ds.select(range(min(len(ds), args.sample)))
    bad_role = bad_type = no_assistant = nameless = 0
    roles = {'system', 'user', 'assistant', 'tool'}
    for row in sample:
        msgs = row['messages']
        if not msgs or not isinstance(msgs[0], dict):
            bad_type += 1
            continue
        if any((m['role'] not in roles for m in msgs)):
            bad_role += 1
        if not any((m['role'] == 'assistant' for m in msgs)):
            no_assistant += 1
        if not all((isinstance(m.get('content'), str) for m in msgs)):
            bad_type += 1
        for tool in row.get('tools') or []:
            if not tool['function']['name']:
                nameless += 1
    rep.check(bad_type == 0, f'content is always str, messages are List[Dict] ({len(sample)} rows)')
    rep.check(bad_role == 0, 'no unknown roles')
    rep.check(no_assistant == 0, 'every row has an assistant turn')
    rep.check(nameless == 0, 'no nameless tool definitions survived')

def load_source_split(report: dict, split: str):
    source = report['source']
    repo, config = (source['repo_id'], source.get('config'))
    if source.get('resolver') == 'datasets':
        return load_plain(repo, config, source.get('revision', 'main'), [split], None)[split]
    plan = resolve_shards(repo, config, source['revision'], [split], allow_stale=False)
    return load_source(repo, plan, None)[split]
FLAT_FORMATS = ('alpaca', 'prompt-completion')
TURN_TEXT_KEYS = ('content', 'value', 'text')

def source_texts(row: dict, report: dict) -> list[str]:
    source = report.get('source', {}) or {}
    columns = source.get('columns') or {}
    fmt = source.get('input_format', 'openai')
    if fmt in FLAT_FORMATS:
        flat = columns.get('flat') or {}
        texts = []
        for key in ('system', 'user', 'input', 'assistant'):
            column = flat.get(key)
            if column and row.get(column) is not None:
                text = str(row[column]).strip()
                if text:
                    texts.append(text)
        return texts
    column = columns.get('messages')
    if not column or column not in row:
        column = next((c for c in ('messages_json', 'messages', 'conversations', 'conversation', 'chat') if c in row), None)
    if column is None:
        return []
    raw = row[column]
    turns = json.loads(raw) if isinstance(raw, str) else raw
    texts = []
    for turn in turns or []:
        if not isinstance(turn, dict):
            continue
        for key in TURN_TEXT_KEYS:
            value = turn.get(key)
            if value:
                texts.append(str(value))
                break
        trace = turn.get('reasoning_content')
        if trace:
            texts.append(str(trace))
    return texts

def check_fidelity(out_dir: Path, rep: Report, args: argparse.Namespace) -> None:
    from datasets import load_dataset
    report = json.loads((out_dir / 'conversion_report.json').read_text())
    split = next(iter(report['splits']))
    src = load_source_split(report, split)
    out = load_dataset(str(out_dir), split=split)
    if len(src) != len(out):
        rep.note(f'source split {split} has {len(src)} rows vs {len(out)} converted (filters or --limit)')
    step = max(1, len(out) // 8)
    picked = out.select(range(0, len(out), step)[:8])
    by_id = 'id' in src.column_names
    if not by_id:
        rep.note('the source has no id column; comparing by position')
    pos: dict[str, int] = {}
    if by_id:
        for i, rid in enumerate(src.data.column('id').to_pylist()):
            pos.setdefault(str(rid), i)
    ok = 0
    problems: list[str] = []
    for offset, row in enumerate(picked):
        if by_id:
            index = pos.get(str(row['id']))
            if index is None:
                problems.append(f"{row['id']}: id absent from the source")
                continue
        else:
            index = min(offset * step, len(src) - 1)
        rendered = '\n'.join((turn['content'] for turn in row['messages']))
        missing = [text for text in source_texts(src[index], report) if text not in rendered]
        if missing:
            problems.append(f"{row['id']}: {len(missing)} source text(s) not preserved, first is {missing[0][:60]!r}")
            continue
        ok += 1
    for problem in problems:
        rep.note(problem)
    rep.check(ok == len(picked), f'{ok}/{len(picked)} sampled rows preserve every source text', otherwise=f'{len(picked) - ok} of {len(picked)} sampled rows lost source text in conversion')

def tool_template(tok) -> dict | None:
    tpl = tok.chat_template
    if isinstance(tpl, dict):
        if 'tools' in (tpl.get('tool_use') or ''):
            return {'chat_template': 'tool_use'}
        tpl = tpl.get('default', '')
    return {} if isinstance(tpl, str) and 'tools' in tpl else None

def check_templates(out_dir: Path, rep: Report, args: argparse.Namespace) -> None:
    from datasets import load_dataset
    from transformers import AutoTokenizer
    splits = load_dataset(str(out_dir))
    ds = splits[next(iter(splits))]
    probe = ds.select(range(min(len(ds), args.sample)))
    plain = tools_row = None
    for i, tl in enumerate(probe['tools'] if 'tools' in ds.features else []):
        if tl and tools_row is None:
            tools_row = probe[i]
        elif not tl and plain is None:
            plain = probe[i]
        if plain and tools_row:
            break
    plain = plain or probe[0]
    if tools_row is None:
        rep.note('no tool-calling row in the probe window; expected when the source carries no tools')
    for name in args.tokenizer or TOKENIZERS:
        try:
            tok = AutoTokenizer.from_pretrained(name)
        except Exception as exc:
            rep.skip(f'{name}: {type(exc).__name__}')
            continue
        text = tok.apply_chat_template(plain['messages'], tokenize=False)
        rep.check(bool(text), f'{name}: plain conversation renders ({len(text)} chars)')
        if tools_row is None:
            continue
        kwargs = tool_template(tok)
        if kwargs is None:
            rep.skip(f'{name}: template ignores `tools=`, nothing to verify')
            continue
        tools = unpack_tools(tools_row['tools'])
        try:
            rendered = tok.apply_chat_template(tools_row['messages'], tokenize=False, tools=tools, **kwargs)
            try:
                bare = tok.apply_chat_template(tools_row['messages'], tokenize=False, **kwargs)
            except Exception:
                bare = None
            names = [t['function']['name'] for t in tools]
            missing = [n for n in names if n not in rendered]
            rep.check(not missing and rendered != bare, f'{name}: all {len(tools)} tool schemas reach the prompt' + (f' (missing {missing})' if missing else ''))
        except Exception as exc:
            rep.check(False, f'{name}: tools render failed: {exc}')
CHECKS = (('schema', 'identical Arrow schema across every shard', check_shard_schemas), ('concat', 'every split concatenates without a CastError', check_concat), ('structure', 'OpenAI message contract', check_structure), ('fidelity', 'byte-level fidelity against the source repo (network)', check_fidelity), ('templates', 'apply_chat_template across tokenizers (network)', check_templates))

def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument('dirs', nargs='*', type=Path, default=[Path('./converted_dataset')], metavar='DIR', help='output directories to verify')
    parser.add_argument('-s', '--skip', action='append', default=[], choices=[name for name, _, _ in CHECKS], metavar='CHECK', help='skip a check; repeatable. one of: ' + ', '.join((name for name, _, _ in CHECKS)))
    parser.add_argument('--sample', type=int, default=3000, metavar='N', help='rows to probe for the structure and template checks')
    parser.add_argument('-t', '--tokenizer', action='append', default=None, metavar='NAME', help=f"tokenizer for the template check; repeatable. default: {', '.join(TOKENIZERS)}")

def run(args: argparse.Namespace) -> int:
    rep = Report()
    for out_dir in args.dirs:
        print(f'verifying {out_dir}')
        for number, (name, title, fn) in enumerate(CHECKS, 1):
            print(f'\n[{number}] {title}')
            if name in args.skip:
                rep.skip(f'{name} skipped on request')
                continue
            try:
                fn(out_dir, rep, args)
            except Exception as exc:
                rep.check(False, f'{out_dir}/{name} crashed: {type(exc).__name__}: {exc}')
    print()
    if rep.fails:
        print(f'{len(rep.fails)} check(s) failed:')
        for fail in rep.fails:
            print(f'  - {fail}')
        return 1
    print('all checks passed')
    return 0
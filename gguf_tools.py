"""Diagnose and repair a fine-tuned model exported to GGUF.

A fine-tune that handled tool calls perfectly in Python often loses the
ability once exported: llama.cpp, Ollama and LM Studio drive tools purely
from the Jinja chat template stored in GGUF metadata under
`tokenizer.chat_template`. Converters routinely write a template that has no
`tools` branch (or write no template at all), and the model then has no way to
be shown a tool schema no matter how well it was trained.

`doctor` reads the metadata and reports what the file can actually do.
`template` writes a Jinja template into a copy of the file, reusing
gguf.scripts.gguf_new_metadata so tensor data is copied verbatim.

Neither command loads model weights, so both are fast on multi-GB files.
"""
from __future__ import annotations
import argparse
from pathlib import Path
from verify_output import Report
HERE = Path(__file__).parent.resolve()
DEFAULT_TEMPLATE = HERE / 'tool_template.jinja'
TOOL_MARKERS = ('tools', 'tool_call')
SAMPLE_TOOLS = [{'type': 'function', 'function': {'name': 'read_file', 'description': 'Read the contents of a file.', 'parameters': {'type': 'object', 'properties': {'path': {'type': 'string'}}, 'required': ['path']}}}, {'type': 'function', 'function': {'name': 'run_python', 'description': 'Execute Python and return stdout.', 'parameters': {'type': 'object', 'properties': {'code': {'type': 'string'}}, 'required': ['code']}}}]
THINK = '<think>\nInspect the file before editing it.\n</think>\n\n'
SAMPLE_MESSAGES = [{'role': 'system', 'content': 'You are a careful workspace assistant.'}, {'role': 'user', 'content': 'Fix the bug in totals.py.'}, {'role': 'assistant', 'content': THINK + 'Reading the file first.', 'tool_calls': [{'id': 'call_1', 'type': 'function', 'function': {'name': 'read_file', 'arguments': '{"path": "src/totals.py"}'}}]}, {'role': 'tool', 'tool_call_id': 'call_1', 'name': 'read_file', 'content': 'def compute(items):\n    return sum(items) + 11'}, {'role': 'assistant', 'content': 'The stray `+ 11` is the bug.'}]

def load_gguf():
    try:
        import gguf
    except ImportError:
        raise SystemExit('the gguf package is required for this command: pip install gguf') from None
    return gguf

def jinja_env():
    from jinja2.sandbox import ImmutableSandboxedEnvironment

    def raise_exception(message: str):
        raise ValueError(message)

    def strftime_now(fmt: str) -> str:
        from datetime import datetime
        return datetime.now().strftime(fmt)
    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
    env.globals['raise_exception'] = raise_exception
    env.globals['strftime_now'] = strftime_now
    return env

def render(template: str, messages: list[dict], tools: list[dict] | None, **extra) -> str:
    tpl = jinja_env().from_string(template)
    return tpl.render(messages=messages, tools=tools, add_generation_prompt=True, **extra)

class Metadata:

    def __init__(self, path: Path) -> None:
        gguf = load_gguf()
        self.path = path
        self.reader = gguf.GGUFReader(path, 'r')
        self.keys = gguf.Keys

    def field(self, key: str):
        field = self.reader.get_field(key)
        return field.contents() if field is not None else None

    @property
    def arch(self) -> str | None:
        return self.field(self.keys.General.ARCHITECTURE)

    @property
    def template(self) -> str | None:
        return self.field(self.keys.Tokenizer.CHAT_TEMPLATE)

    @property
    def named_templates(self) -> list[str]:
        prefix = self.keys.Tokenizer.CHAT_TEMPLATE + '.'
        return sorted((name for name in self.reader.fields if name.startswith(prefix)))

    @property
    def tokens(self) -> list[str]:
        return self.field(self.keys.Tokenizer.LIST) or []

    def token(self, key: str) -> tuple[int | None, str | None]:
        tid = self.field(key)
        if tid is None:
            return (None, None)
        tokens = self.tokens
        text = tokens[tid] if 0 <= tid < len(tokens) else None
        return (tid, text)

def sample_conversation(args: argparse.Namespace, rep: Report) -> tuple[list[dict], list[dict]]:
    if not getattr(args, 'dataset', None):
        return (SAMPLE_MESSAGES, SAMPLE_TOOLS)
    from datasets import load_dataset
    from convert_dataset import unpack_tools
    splits = load_dataset(str(args.dataset))
    ds = splits[next(iter(splits))]
    probe = ds.select(range(min(len(ds), args.sample)))
    column = probe['tools'] if 'tools' in ds.features else []
    for i, tools in enumerate(column):
        if tools:
            rep.note(f"probing with row {probe[i]['id']} from {args.dataset}")
            return (probe[i]['messages'], unpack_tools(tools))
    rep.note(f'no tool-calling row in the first {len(probe)} rows of {args.dataset} -- using the built-in sample')
    return (SAMPLE_MESSAGES, SAMPLE_TOOLS)
ENDIAN = {'I': 'little', 'S': 'big'}

def check_file(meta: Metadata, rep: Report, args: argparse.Namespace) -> None:
    from push_to_hf import human
    order = ENDIAN.get(meta.reader.byte_order, meta.reader.byte_order)
    rep.check(meta.arch is not None, f'architecture {meta.arch!r}, {len(meta.reader.tensors)} tensors, {human(meta.path.stat().st_size)}, {order}-endian')

def check_template(meta: Metadata, rep: Report, args: argparse.Namespace) -> None:
    template = meta.template
    if template:
        rep.check(True, f'tokenizer.chat_template is set ({len(template)} chars)')
    else:
        rep.check(False, 'tokenizer.chat_template is missing -- the runtime falls back to a generic prompt and tools are unreachable')
    for name in meta.named_templates:
        rep.note(f'additional template stored under {name}')

def check_tools(meta: Metadata, rep: Report, args: argparse.Namespace) -> None:
    template = meta.template or ''
    hits = [marker for marker in TOOL_MARKERS if marker in template]
    if hits:
        rep.check(True, f"template references tool calling (matched {', '.join(hits)})")
    else:
        rep.check(False, 'no tool-calling schema in the template -- the model cannot be shown a tool no matter how it was trained')

def check_render(meta: Metadata, rep: Report, args: argparse.Namespace) -> None:
    template = meta.template
    if not template:
        rep.skip('no template to render')
        return
    messages, tools = sample_conversation(args, rep)
    try:
        prompt = render(template, messages, tools)
    except Exception as exc:
        rep.check(False, f'rendering with tools raised {type(exc).__name__}: {exc}')
        return
    try:
        bare = render(template, messages, None)
    except Exception:
        bare = None
    rep.check(prompt != bare, 'passing tools changes the prompt', otherwise='the tools argument does not change the prompt at all -- the template accepts it and ignores it')
    names = [t['function']['name'] for t in tools]
    missing = [n for n in names if n not in prompt]
    rep.check(not missing, f'all {len(names)} tool schemas reach the prompt', otherwise=f"{len(missing)} of {len(names)} tool schemas never reach the prompt: {', '.join(missing)}")
    calls = [c['function']['name'] for m in messages for c in m.get('tool_calls') or []]
    lost = [n for n in calls if n not in prompt]
    if calls:
        rep.check(not lost, f'{len(calls)} assistant tool call(s) survive the template', otherwise=f"{len(lost)} assistant tool call(s) are dropped by the template: {', '.join(lost)}")
    results = [m['content'] for m in messages if m['role'] == 'tool' and m.get('content')]
    dropped = [r for r in results if r[:40] not in prompt]
    if results:
        rep.check(not dropped, f'{len(results)} tool result(s) are fed back into the conversation', otherwise=f'{len(dropped)} of {len(results)} tool result(s) are dropped, so the model never sees what a tool returned')

def check_stop(meta: Metadata, rep: Report, args: argparse.Namespace) -> None:
    eos_id, eos = meta.token(meta.keys.Tokenizer.EOS_ID)
    eot_id, eot = meta.token(meta.keys.Tokenizer.EOT_ID)
    if eos_id is None:
        rep.check(False, 'no eos_token_id -- generation will not stop by itself')
    else:
        rep.check(eos is not None, f'eos_token_id {eos_id} = {eos!r}' if eos is not None else f'eos_token_id {eos_id} is outside the token list')
    if eot_id is not None:
        rep.note(f'eot_token_id {eot_id} = {eot!r}')
    template = meta.template
    if not template:
        rep.skip('no template to check the turn terminator against')
        return
    try:
        prompt = render(template, SAMPLE_MESSAGES, SAMPLE_TOOLS)
    except Exception as exc:
        rep.skip(f'template does not render: {type(exc).__name__}: {exc}')
        return
    stops = [tok for tok in (eos, eot) if tok]
    if not stops:
        return
    rep.check(any((tok in prompt for tok in stops)), f"the template ends turns with a stop token ({', '.join(stops)})", otherwise=f"the template's turn terminator is not the model's stop token ({', '.join(stops)}) -- generation will run past the end of a turn")

def check_reasoning(meta: Metadata, rep: Report, args: argparse.Namespace) -> None:
    template = meta.template
    if not template:
        rep.skip('no template to render')
        return
    try:
        prompt = render(template, SAMPLE_MESSAGES, SAMPLE_TOOLS)
    except Exception as exc:
        rep.skip(f'template does not render: {type(exc).__name__}: {exc}')
        return
    if '<think>' in prompt:
        rep.note('<think> blocks in the history are passed through verbatim')
    else:
        rep.note('<think> blocks are stripped from the history; normal for Qwen3-style templates, but the training rows carry them inline, so served history differs from training')
CHECKS = (('file', 'GGUF header and tensor inventory', check_file), ('template', 'tokenizer.chat_template is present', check_template), ('tools', 'the template carries a tool-calling schema', check_tools), ('render', 'a tool-calling conversation renders', check_render), ('stop', 'stop tokens agree with the template', check_stop), ('reasoning', 'reasoning traces in the history', check_reasoning))
TEMPLATE_CHECKS = {'template', 'tools', 'render'}

def add_probe_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument('-d', '--dataset', type=Path, default=None, metavar='DIR', help='converted output directory to draw a real tool-calling row from, instead of the built-in sample conversation')
    parser.add_argument('--sample', type=int, default=2000, metavar='N', help='rows to scan in DIR for a tool-calling row')

def add_doctor_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument('models', nargs='+', type=Path, metavar='MODEL.gguf', help='exported model files to check')
    parser.add_argument('-s', '--skip', action='append', default=[], choices=[name for name, _, _ in CHECKS], metavar='CHECK', help='skip a check; repeatable. one of: ' + ', '.join((name for name, _, _ in CHECKS)))
    add_probe_arguments(parser)

def run_doctor(args: argparse.Namespace) -> int:
    rep = Report()
    for path in args.models:
        print(f'doctor {path}')
        if not path.is_file():
            rep.check(False, f'{path} does not exist')
            continue
        try:
            meta = Metadata(path)
        except Exception as exc:
            rep.check(False, f'{path} does not read as GGUF: {type(exc).__name__}: {exc}')
            continue
        for number, (name, title, fn) in enumerate(CHECKS, 1):
            print(f'\n[{number}] {title}')
            if name in args.skip:
                rep.skip(f'{name} skipped on request')
                continue
            try:
                fn(meta, rep, args)
            except Exception as exc:
                rep.check(False, f'{name} crashed: {type(exc).__name__}: {exc}')
    print()
    if not rep.fails:
        print('no problems found')
        return 0
    print(f'{len(rep.fails)} problem(s) found:')
    for fail in rep.fails:
        print(f'  - {fail}')
    if any((marker in fail for fail in rep.fails for marker in ('chat_template', 'tool'))):
        print('\nthe template is repairable without retraining: write one into a copy of the file with the template command, then re-run doctor on the copy')
    return 1

def add_template_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument('input', type=Path, metavar='IN.gguf', help='exported model to read')
    parser.add_argument('output', type=Path, metavar='OUT.gguf', help='copy to write, with the template attached')
    parser.add_argument('-f', '--template-file', type=Path, default=DEFAULT_TEMPLATE, metavar='FILE', help='Jinja chat template to store under tokenizer.chat_template')
    parser.add_argument('--force', action='store_true', help='overwrite OUT.gguf if it already exists')
    parser.add_argument('--no-check', action='store_true', help='do not run the doctor checks over the result')
    add_probe_arguments(parser)

def run_template(args: argparse.Namespace) -> int:
    gguf = load_gguf()
    from gguf.scripts.gguf_new_metadata import MetadataDetails, copy_with_new_metadata
    if not args.template_file.is_file():
        raise SystemExit(f'no template file at {args.template_file}')
    template = args.template_file.read_text(encoding='utf-8')
    try:
        prompt = render(template, SAMPLE_MESSAGES, SAMPLE_TOOLS)
    except Exception as exc:
        raise SystemExit(f'{args.template_file} does not render: {type(exc).__name__}: {exc}') from None
    missing = [t['function']['name'] for t in SAMPLE_TOOLS if t['function']['name'] not in prompt]
    if missing:
        raise SystemExit(f'{args.template_file} renders but drops the tool schemas {missing} -- it would not fix tool calling')
    print(f'template {args.template_file} renders, {len(SAMPLE_TOOLS)} tool schemas reach the prompt')
    if not args.input.is_file():
        raise SystemExit(f'no such file: {args.input}')
    if args.output.exists() and (not args.force):
        raise SystemExit(f'{args.output} exists (pass --force to overwrite)')
    if args.output.resolve() == args.input.resolve():
        raise SystemExit('in-place rewriting is not supported; write to a different path')
    reader = gguf.GGUFReader(args.input, 'r')
    arch_field = reader.get_field(gguf.Keys.General.ARCHITECTURE)
    if arch_field is None:
        raise SystemExit(f'{args.input} has no general.architecture field')
    writer = gguf.GGUFWriter(args.output, arch=arch_field.contents(), endianess=reader.endianess)
    alignment = reader.get_field(gguf.Keys.General.ALIGNMENT)
    if alignment is not None:
        writer.data_alignment = alignment.contents()
    new = {gguf.Keys.Tokenizer.CHAT_TEMPLATE: MetadataDetails(gguf.GGUFValueType.STRING, template)}
    copy_with_new_metadata(reader, writer, new, [])
    print(f'wrote {args.output}')
    if args.no_check:
        return 0
    print()
    probe = argparse.Namespace(models=[args.output], skip=[], dataset=args.dataset, sample=args.sample)
    return run_doctor(probe)
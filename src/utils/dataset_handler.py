import json
import re
from collections import namedtuple
from datetime import datetime, timezone
from pathlib import Path

from datasets import concatenate_datasets, load_dataset

from .setup import PROJECT_ROOT
from .llm_client import call_llm
from .lean_validator import new_server, validate_lean


_DATA_DIR = PROJECT_ROOT / "data"  # gitignored HF cache

_COLUMNS = ["header", "statement", "proof", "text"]


# adapt: raw row -> dict with exactly the keys in _COLUMNS (all str)
_Spec = namedtuple("_Spec", "repo role split config adapt")


_SEED = 0
_STP_FENCE = "```lean4"
_PROOF_SEP = ":= by"
# lines that form the Lean preamble (before the first declaration)
_HEADER_PREFIXES = ("import ", "open ", "set_option ", "variable",
                    "namespace ", "section", "universe", "noncomputable section")


def _join(*parts):
    return "\n".join(p.strip("\n") for p in parts if p)


def _split_header(blob):
    # header = leading run of import/open/set_option/... lines; rest is the body
    lines = blob.split("\n")
    i = 0
    while i < len(lines) and (not lines[i].strip()
                              or lines[i].lstrip().startswith(_HEADER_PREFIXES)):
        i += 1
    return _join("\n".join(lines[:i])), _join("\n".join(lines[i:]))


def _split_proof(body):
    # split the declaration signature (through ':= by') from the tactic proof
    idx = body.find(_PROOF_SEP)
    if idx == -1:
        return _join(body), ""
    cut = idx + len(_PROOF_SEP)
    return _join(body[:cut]), _join(body[cut:])


def _adapt_deepseek(row):
    header = row.get("header") or ""
    statement = row.get("formal_statement") or ""
    proof = row.get("formal_proof") or ""
    return {"header": _join(header), "statement": _join(statement),
            "proof": _join(proof), "text": _join(header, statement, proof)}


def _adapt_goedel(row):
    # full_proof bundles header + statement + proof in one blob; split it out.
    header, body = _split_header(row.get("full_proof") or "")
    statement, proof = _split_proof(body)
    return {"header": header, "statement": statement, "proof": proof,
            "text": _join(header, statement, proof)}


def _adapt_stp(row):
    # prompt = "Complete the following Lean 4 code:\n\n```lean4\n<header+statement>"
    code = row.get("prompt") or ""
    if _STP_FENCE in code:
        code = code.split(_STP_FENCE, 1)[1]
    code = code.strip("\n")
    if code.endswith("```"):
        code = code[:-3]
    header, statement = _split_header(code)
    proof = _join(row.get("target") or "")
    return {"header": header, "statement": statement, "proof": proof,
            "text": _join(header, statement, proof)}


def _adapt_statement_only(row):
    # eval benchmarks ship statement + header but no formal proof
    header = row.get("header") or ""
    statement = row.get("formal_statement") or ""
    return {"header": header, "statement": statement, "proof": "",
            "text": _join(header, statement)}


_DATASETS = {
    # --- whole-proof training corpora ---
    "deepseek-prover-v1": _Spec("deepseek-ai/DeepSeek-Prover-V1",
                                "train", "train", None, _adapt_deepseek),
    "lean-workbook-proofs": _Spec("Goedel-LM/Lean-workbook-proofs",
                                  "train", "train", None, _adapt_goedel),
    "stp-lean": _Spec("kfdong/STP_Lean_0320",
                      "train", "train", None, _adapt_stp),
    # --- held-out eval benchmarks (never reachable via load_train_dataset) ---
    "minif2f": _Spec("cat-searcher/minif2f-lean4",
                     "eval", "test", None, _adapt_statement_only),
    "proofnet": _Spec("UDACA/proofnet-lean4",
                      "eval", "test", None, _adapt_statement_only),
}


def list_datasets(role=None):
    return [name for name, spec in _DATASETS.items()
            if role is None or spec.role == role]


def _load(name, expected_role, **kwargs):
    if name is None:
        names = list_datasets(expected_role)
        combined = concatenate_datasets([_load(n, expected_role, **kwargs) for n in names])
        return combined.shuffle(seed=_SEED) if expected_role == "train" else combined
    if name not in _DATASETS:
        raise KeyError(f"unknown dataset {name!r}; choose from {list(_DATASETS)}")
    spec = _DATASETS[name]
    if spec.role != expected_role:
        raise ValueError(
            f"{name!r} is a {spec.role!r} dataset; "
            f"use load_{'eval_benchmark' if spec.role == 'eval' else 'train_dataset'}()")

    ds = load_dataset(spec.repo, name=spec.config, split=spec.split,
                      cache_dir=str(_DATA_DIR), **kwargs)
    return ds.map(spec.adapt, remove_columns=ds.column_names)


def load_train_dataset(name=None, **kwargs):
    return _load(name, "train", **kwargs)


def load_eval_benchmark(name, **kwargs):
    return _load(name, "eval", **kwargs)


# --- Isabelle AFP -> Lean 4 transpilation ---

_LEAN_MANIFEST_NAME = "manifest.json"

_SESSION_RE = re.compile(r'session\s+"?([\w\-]+)"?\s*=')
_COMMENT_RE = re.compile(r'\(\*.*?\*\)', re.S)
_IMPORTS_RE = re.compile(r'^\s*imports\b(.*?)\bbegin\b', re.S | re.M)
_TOKEN_RE = re.compile(r'"([^"]+)"|(\S+)')

_SYSTEM_PROMPT = (
    "You are an expert in both Isabelle/HOL and Lean 4 (with Mathlib). Translate the "
    "given Isabelle theory source into a single self-contained Lean 4 file that "
    "type-checks, preserving the mathematical content and proofs as faithfully as "
    "possible. Reply with only the Lean 4 code, no commentary, no markdown fences."
)
_LEAN_FENCE = "```lean4"


def _tokenize(block):
    return [a or b for a, b in _TOKEN_RE.findall(block)]


def _normalize(name):
    return name.replace("-", "_").lower()


def _parse_root(root_path):
    text = _COMMENT_RE.sub(" ", root_path.read_text(encoding="utf-8", errors="replace"))
    m = _SESSION_RE.search(text)
    if not m:
        raise ValueError(f"no session declaration found in {root_path}")
    return m.group(1)


def _find_main_theory(entry_dir, session_name):
    target = _normalize(session_name)
    candidates = [p for p in entry_dir.rglob("*.thy") if _normalize(p.stem) == target]
    if not candidates:
        return None
    return candidates[0]


def _parse_imports(thy_path):
    text = _COMMENT_RE.sub(" ", thy_path.read_text(encoding="utf-8", errors="replace"))
    m = _IMPORTS_RE.search(text)
    return _tokenize(m.group(1)) if m else []


def _build_translation_unit(entry_dir, main_thy_path):
    thy_by_stem = {p.stem: p for p in entry_dir.rglob("*.thy")}
    order, visited = [], set()

    def visit(path):
        if path in visited:
            return
        visited.add(path)
        for tok in _parse_imports(path):
            if "." in tok or "/" in tok:
                continue
            dep = thy_by_stem.get(tok)
            if dep is not None and dep != path:
                visit(dep)
        order.append(path)

    visit(main_thy_path)
    return order


def _assemble_source(order):
    return "\n\n".join(
        f"(* --- {p.name} --- *)\n{p.read_text(encoding='utf-8', errors='replace')}"
        for p in order
    )


def _strip_fence(code):
    if _LEAN_FENCE in code:
        code = code.split(_LEAN_FENCE, 1)[1]
    elif "```" in code:
        code = code.split("```", 1)[1]
    return code.split("```")[0].strip("\n")


def _build_retry_prompt(source_block, prev_attempt, error_text):
    # source_block carries the same cache_control mark on every retry for this
    # entry -- only this function's second block varies, so the cached prefix
    # (system prompt + source_block) is re-read from cache, not re-written.
    return [
        source_block,
        {"type": "text", "text": (
            f"Previous Lean 4 attempt:\n```lean4\n{prev_attempt}\n```\n\n"
            f"Pantograph compiler error:\n```\n{error_text}\n```\n\n"
            "Your previous attempt failed to type-check. Fix the errors shown and "
            "produce a corrected, complete Lean 4 file.")},
    ]


def _transpile_entry(source, source_language, provider, model, max_retries):
    server = new_server()
    source_block = {
        "type": "text",
        "text": f"Theory source ({source_language}):\n```{source_language}\n{source}\n```",
        "cache_control": {"type": "ephemeral"},
    }
    user_prompt = [source_block]
    lean_code, err, attempt = "", "", 0
    for attempt in range(1, max_retries + 2):  # 1 initial + up to max_retries retries
        lean_code = _strip_fence(call_llm(_SYSTEM_PROMPT, user_prompt, provider=provider, model=model,
                                           cache_system=True))
        ok, err = validate_lean(server, lean_code)
        if ok:
            return lean_code, True, "", attempt
        user_prompt = _build_retry_prompt(source_block, lean_code, err)
    return lean_code, False, err, attempt


def _now():
    return datetime.now(timezone.utc).isoformat()


def _load_manifest(manifest_path):
    return json.loads(manifest_path.read_text()) if manifest_path.exists() else {}


def _save_manifest(manifest_path, manifest):
    tmp = manifest_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    tmp.replace(manifest_path)


def _iter_entries(dataset_path):
    for entry_dir in sorted(Path(dataset_path).iterdir()):
        if entry_dir.is_dir() and (entry_dir / "ROOT").exists():
            yield entry_dir


def transpile_to_lean(dataset_path, source_language="isabelle", *,
                       out_dir=None, manifest_path=None, max_retries=3,
                       provider=None, model=None):
    """Transpile each Formal Proof entry under dataset_path (e.g. data/isa___afp/thys) to Lean 4.

    Assumes setup() has already been called by the caller so API keys are in
    os.environ.
    """
    dataset_path = Path(dataset_path)
    out_dir = Path(out_dir) if out_dir else _DATA_DIR / f"{dataset_path.parent.name}_lean"
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = Path(manifest_path) if manifest_path else out_dir / _LEAN_MANIFEST_NAME
    manifest = _load_manifest(manifest_path)

    n_success, n_attempted = 0, 0
    for entry_dir in _iter_entries(dataset_path):
        name = entry_dir.name
        if name in manifest:
            continue
        n_attempted += 1
        try:
            session_name = _parse_root(entry_dir / "ROOT")
            main_thy = _find_main_theory(entry_dir, session_name)
            if main_thy is None:
                raise ValueError("no .thy file matches session name")
            order = _build_translation_unit(entry_dir, main_thy)
            source = _assemble_source(order)
        except Exception as exc:
            manifest[name] = {"status": "dropped", "attempts": 0, "timestamp": _now(),
                              "output_path": None, "error": f"parse error: {exc}"}
            _save_manifest(manifest_path, manifest)
            continue

        lean_code, ok, err, attempts = _transpile_entry(
            source, source_language, provider, model, max_retries)

        if ok:
            entry_out = out_dir / f"{name}_converted"
            entry_out.mkdir(parents=True, exist_ok=True)
            lean_path = entry_out / f"{main_thy.stem}.lean"
            lean_path.write_text(lean_code)
            (entry_out / "source.isa.txt").write_text(source)
            (entry_out / "meta.json").write_text(json.dumps({
                "entry": name, "session_name": session_name,
                "source_language": source_language,
                "files_merged": [p.name for p in order],
            }, indent=2))
            manifest[name] = {"status": "success", "attempts": attempts, "timestamp": _now(),
                              "output_path": str(lean_path.relative_to(_DATA_DIR)), "error": None}
            n_success += 1
        else:
            print(f"[transpile_to_lean] dropped {name} after {attempts} attempts: "
                  f"{err.splitlines()[0] if err else '(no error text)'}")
            manifest[name] = {"status": "dropped", "attempts": attempts, "timestamp": _now(),
                              "output_path": None, "error": err}

        _save_manifest(manifest_path, manifest)

    cumulative_success = sum(1 for v in manifest.values() if v["status"] == "success")
    print(f"transpile_to_lean: {n_success}/{n_attempted} succeeded this run "
          f"({cumulative_success}/{len(manifest)} succeeded cumulatively)")
    return manifest

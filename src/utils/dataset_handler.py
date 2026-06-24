from collections import namedtuple

from datasets import concatenate_datasets, load_dataset

from .setup import PROJECT_ROOT


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

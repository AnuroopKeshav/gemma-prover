import threading
from concurrent.futures import ThreadPoolExecutor

from .setup import PROJECT_ROOT

# Lean project with Mathlib built, pinned to the Lean toolchain pantograph's
# bundled REPL binary was compiled against -- see out/lean-toolchain.
_LEAN_PROJECT_PATH = str(PROJECT_ROOT / "out")

# pantograph's sync wrappers bind an asyncio event loop at import time and
# call loop.run_until_complete() on every operation. That's incompatible with
# environments that already run a loop (e.g. Jupyter/ipykernel), so all
# pantograph work is funneled through one dedicated worker thread with its
# own fresh loop instead of patching/sharing the caller's loop.
_executor = None
_executor_lock = threading.Lock()


def _get_executor():
    global _executor
    with _executor_lock:
        if _executor is None:
            _executor = ThreadPoolExecutor(max_workers=1)
            _executor.submit(__import__, "pantograph").result()
        return _executor


def new_server(**kwargs):
    kwargs.setdefault("imports", ["Mathlib"])
    kwargs.setdefault("project_path", _LEAN_PROJECT_PATH)

    def _build():
        import pantograph
        return pantograph.Server(**kwargs)
    return _get_executor().submit(_build).result()


def _strip_leading_imports(code):
    # pantograph's Server preloads `imports` at construction, so the snippet
    # handed to load_sorry must not repeat its own `import` lines -- from the
    # kernel's view those aren't "at the beginning of the file" since the real
    # beginning was already consumed by server startup.
    lines = code.split("\n")
    i = 0
    while i < len(lines) and (not lines[i].strip() or lines[i].startswith("import ")):
        i += 1
    return "\n".join(lines[i:])


def validate_lean(server, code):
    body = _strip_leading_imports(code)

    def _run():
        try:
            units = server.load_sorry(body)
        except Exception as exc:
            return False, str(exc)
        errors = [str(msg) for unit in units for msg in getattr(unit, "messages", [])
                  if getattr(msg, "level", "error") == "error"]
        return (False, "\n".join(errors)) if errors else (True, "")

    return _get_executor().submit(_run).result()


_SANITY_GOOD = "theorem sanity_check_ok (n : Nat) : n + 0 = n := by simp"
_SANITY_BAD = "theorem sanity_check_bad (n : Nat) : n + 0 = n + 1 := by simp"


def check_pipeline(**server_kwargs):
    """Sanity-check pantograph/lean setup before spending LLM credits.

    Raises RuntimeError with details if the server can't be built, or if a
    trivially true/false theorem isn't validated correctly.
    """
    try:
        server = new_server(**server_kwargs)
    except Exception as exc:
        raise RuntimeError(f"pantograph server failed to start: {exc}") from exc

    ok, err = validate_lean(server, _SANITY_GOOD)
    if not ok:
        raise RuntimeError(f"known-good Lean snippet failed to validate: {err}")

    ok, err = validate_lean(server, _SANITY_BAD)
    if ok:
        raise RuntimeError("known-bad Lean snippet validated as correct (validator not catching errors)")

    return server

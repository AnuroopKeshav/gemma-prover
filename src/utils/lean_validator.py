def new_server(**kwargs):
    import pantograph
    return pantograph.Server(**kwargs)


def validate_lean(server, code):
    try:
        units = server.load_sorry(code)
    except Exception as exc:
        return False, str(exc)
    errors = [str(msg) for unit in units for msg in getattr(unit, "messages", [])
              if getattr(msg, "level", "error") == "error"]
    return (False, "\n".join(errors)) if errors else (True, "")


_SANITY_GOOD = "theorem sanity_check_ok (n : Nat) : n + 0 = n := by simp"
_SANITY_BAD = "theorem sanity_check_bad (n : Nat) : n + 0 = n + 1 := by simp"


# sanity checks
def check_pipeline(**server_kwargs):
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

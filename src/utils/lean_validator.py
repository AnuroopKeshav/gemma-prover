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

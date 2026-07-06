import os

_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


def _is_reasoning_model():
    return os.environ.get("REASONING_MODEL", "").strip().lower() in ("1", "true", "yes")


def call_llm(system, user, *, provider=None, model=None, temperature=0.0, max_tokens=8192,
             cache_system=False):
    provider = provider or os.environ["API_PROVIDER"]
    model = model or os.environ["MODEL_NAME"]
    if _is_reasoning_model():
        temperature = 1.0
    if provider == "openai":
        return _chat_completions(system, user, model, temperature, max_tokens,
                                  api_key=os.environ["OPENAI_API_KEY"], base_url=None)
    if provider == "openrouter":
        return _chat_completions(system, user, model, temperature, max_tokens,
                                  api_key=os.environ["OPENROUTER_API_KEY"], base_url=_OPENROUTER_BASE_URL)
    if provider == "anthropic":
        return _anthropic_messages(system, user, model, temperature, max_tokens,
                                    api_key=os.environ["ANTHROPIC_API_KEY"], cache_system=cache_system)
    raise ValueError(f"unknown API_PROVIDER {provider!r}; choose openai/anthropic/openrouter")


def _flatten(user):
    if isinstance(user, str):
        return user
    return "\n\n".join(block["text"] for block in user)


def _chat_completions(system, user, model, temperature, max_tokens, api_key, base_url):
    from openai import OpenAI
    client = OpenAI(api_key=api_key, base_url=base_url) if base_url else OpenAI(api_key=api_key)
    resp = client.chat.completions.create(
        model=model, temperature=temperature,
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": _flatten(user)}])
    return resp.choices[0].message.content


def _anthropic_messages(system, user, model, temperature, max_tokens, api_key, cache_system):
    import anthropic
    client = anthropic.Anthropic(api_key=api_key)
    system_param = ([{"type": "text", "text": system,
                       "cache_control": {"type": "ephemeral"}}]
                     if cache_system else system)
    # `user` may be a plain string or a list of Anthropic content blocks (with
    # cache_control on the stable prefix) -- the SDK accepts both as `content`.
    resp = client.messages.create(
        model=model, max_tokens=max_tokens, temperature=temperature,
        system=system_param, messages=[{"role": "user", "content": user}])
    return "".join(b.text for b in resp.content if getattr(b, "type", None) == "text")

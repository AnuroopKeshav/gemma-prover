import torch
import transformers
from transformers import AutoConfig, AutoModel


_LABEL_COL = 110
_PARAM_COL = 14
_TABLE_WIDTH = _LABEL_COL + _PARAM_COL


def _fmt_params(n):
    for unit in ["", "K", "M", "B", "T"]:
        if abs(n) < 1000:
            return f"{n:.2f}{unit}" if unit else str(n)
        n /= 1000
    return f"{n:.2f}P"


def _shape_repr(module):
    parts = [f"{p_name} {tuple(p.shape)}" for p_name, p in module.named_parameters(recurse=False)]
    return ", ".join(parts)


def _print_module(name, module, depth, max_depth, indent):
    n_params = sum(p.numel() for p in module.parameters())
    extra = module.extra_repr() or _shape_repr(module)
    type_name = type(module).__name__
    detail = f" [{extra}]" if extra else ""
    label = f"{'  ' * indent}{name} ({type_name}){detail}"
    print(f"{label:<{_LABEL_COL}}{_fmt_params(n_params):>{_PARAM_COL}}")

    if max_depth is not None and depth >= max_depth:
        return

    children = list(module.named_children())

    if isinstance(module, torch.nn.ModuleList) and children:
        types = {type(child).__name__ for _, child in children}
        if len(types) == 1 and len(children) > 1:
            first_name, first = children[0]
            print(f"{'  ' * (indent + 1)}({first_name}-{children[-1][0]}): "
                  f"{len(children)} x {type(first).__name__}")
            _print_module(first_name, first, depth + 1, max_depth, indent + 1)
            return

    for child_name, child in children:
        _print_module(child_name, child, depth + 1, max_depth, indent + 1)


def _collect_attn_impls(config, _seen=None):
    if _seen is None:
        _seen = set()
    if id(config) in _seen:
        return set()
    _seen.add(id(config))

    impls = set()
    impl = getattr(config, "_attn_implementation", None)
    if impl:
        impls.add(impl)
    for value in vars(config).values():
        if isinstance(value, transformers.PretrainedConfig):
            impls |= _collect_attn_impls(value, _seen)
    return impls


def _force_eager_attention(config, _seen=None):
    if _seen is None:
        _seen = set()
    if id(config) in _seen:
        return
    _seen.add(id(config))

    config._attn_implementation = "eager"
    for value in vars(config).values():
        if isinstance(value, transformers.PretrainedConfig):
            _force_eager_attention(value, _seen)


def _build_meta_model(config, trust_remote_code):
    architectures = getattr(config, "architectures", None) or []
    _force_eager_attention(config)

    with torch.device("meta"):
        if not trust_remote_code:
            for arch in architectures:
                cls = getattr(transformers, arch, None)
                if cls is not None:
                    return cls(config)
        return AutoModel.from_config(config, trust_remote_code=trust_remote_code)


def look_at(source, max_depth=6, trust_remote_code=False):
    config = AutoConfig.from_pretrained(source, trust_remote_code=trust_remote_code)

    requested_attn = ", ".join(sorted(_collect_attn_impls(config))) or "default"

    model = _build_meta_model(config, trust_remote_code)

    total = sum(p.numel() for p in model.parameters())

    text_config = getattr(config, "text_config", config)
    activation = getattr(text_config, "hidden_act", None) or getattr(text_config, "hidden_activation", "?")

    width = _TABLE_WIDTH
    print("=" * width)
    print(f"Model      : {source}")
    print(f"Type       : {getattr(config, 'model_type', '?')} / {type(model).__name__}")
    print(f"Attn impl  : {requested_attn} (forced eager for inspection)")
    print(f"Activation : {activation}")
    print("=" * width)
    print(f"{'Layer (type) [dims]':<{_LABEL_COL}}{'Params':>{_PARAM_COL}}")
    print("-" * width)
    _print_module(type(model).__name__, model, 0, max_depth, 0)
    print("=" * width)
    print(f"{'Total params':<{_LABEL_COL}}{_fmt_params(total):>{_PARAM_COL}}")
    print(f"{'Total params (raw)':<{_LABEL_COL}}{total:>{_PARAM_COL},}")
    print("=" * width)

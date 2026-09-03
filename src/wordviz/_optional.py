import importlib.util

_OPTIONAL_DEPENDENCIES = {
    "viz": {
        "modules": ["umap"],
    },
    "encoding": {
        "modules": ["torch", "transformers", "sentence_transformers"],
    },
}


def require(extra: str) -> None:
    if extra not in _OPTIONAL_DEPENDENCIES:
        raise ValueError(f"Unknown optional dependency group: {extra}")

    modules = _OPTIONAL_DEPENDENCIES[extra]["modules"]

    missing = [m for m in modules if importlib.util.find_spec(m) is None]

    if missing:
        raise ImportError(
            f"Missing optional dependencies: {', '.join(missing)}.\n\n"
            f"Install them with:\n"
            f"    pip install wordviz[{extra}]"
        )

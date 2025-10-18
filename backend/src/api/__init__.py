from typing import Any

__all__ = ["app"]


def __getattr__(name: str) -> Any:
    if name == "app":
        from .main import app as _app

        return _app
    raise AttributeError(name)

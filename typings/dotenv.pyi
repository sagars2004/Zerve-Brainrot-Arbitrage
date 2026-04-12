"""Stubs for python-dotenv (Pylance/basedpyright); install `python-dotenv` to run code."""

from __future__ import annotations

from pathlib import Path
from typing import IO, Optional, Union

StrPath = Union[str, Path]

def load_dotenv(
    dotenv_path: Optional[StrPath] = None,
    stream: Optional[IO[str]] = None,
    *,
    verbose: bool = False,
    override: bool = False,
    interpolate: bool = True,
    encoding: str = "utf-8",
) -> bool: ...

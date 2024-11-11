from __future__ import annotations

import base64
import inspect
import json
from collections.abc import Iterable, Sequence
from functools import partial
from typing import (
    TYPE_CHECKING,
    Annotated,
    Any,
    Callable,
    Literal,
    Optional,
    Union,
    cast,
    overload,
)

from pydantic import Discriminator, Field, Tag

from gwenflow.types.chat import ChatMessage


def _bytes_to_b64_str(bytes_: bytes) -> str:
    return base64.b64encode(bytes_).decode("utf-8")


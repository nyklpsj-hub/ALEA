"""Checkpointer serialisation policy.

LangGraph stores graph state as msgpack. Any custom class in the state has to be
declared, otherwise the deserialiser falls back to a permissive path that warns
today and will be blocked in a future release.

Rather than hand-listing the types (which rots the moment a field is added), the
allowed set is derived from the modules that actually define the state: the
Pydantic contracts and the domain enums. Only those are permitted — everything
else keeps the strict default.
"""

from __future__ import annotations

import inspect
from enum import StrEnum
from functools import lru_cache

from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from pydantic import BaseModel

from app.db import enums as db_enums
from app.schemas import agent as agent_schemas

#: Modules whose classes may appear inside a checkpointed graph state.
STATE_MODULES = (agent_schemas, db_enums)


@lru_cache
def allowed_state_types() -> list[tuple[str, str]]:
    allowed: list[tuple[str, str]] = []
    for module in STATE_MODULES:
        for name, obj in vars(module).items():
            if not inspect.isclass(obj) or getattr(obj, "__module__", None) != module.__name__:
                continue
            if issubclass(obj, (BaseModel, StrEnum)):
                allowed.append((module.__name__, name))
    return sorted(set(allowed))


def build_serde() -> JsonPlusSerializer:
    return JsonPlusSerializer(allowed_msgpack_modules=allowed_state_types())

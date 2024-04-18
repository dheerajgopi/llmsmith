from dataclasses import dataclass
from typing import Any, Generic, TypeVar


T = TypeVar("T")


@dataclass
class TaskInput(Generic[T]):
    content: T


@dataclass
class TaskOutput(Generic[T]):
    content: T
    raw_output: Any


@dataclass
class ChatResponse:
    text: str
    raw_output: Any
    function_calls: dict[str, dict[str, Any]] = None

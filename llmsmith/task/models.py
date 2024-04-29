from dataclasses import dataclass
from typing import Any, Generic, TypeVar, Union


T = TypeVar("T")


@dataclass
class TaskInput(Generic[T]):
    content: T


@dataclass
class TaskOutput(Generic[T]):
    content: T
    raw_output: Any


@dataclass
class FunctionCall:
    id: Union[str, None]
    args: dict[str, Any]


@dataclass
class ChatResponse:
    text: str
    raw_output: Any
    function_calls: dict[str, FunctionCall] = None

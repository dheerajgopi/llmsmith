from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from llmsmith.task.models import TaskInput, TaskOutput


T = TypeVar("T")
U = TypeVar("U")


class Task(Generic[T, U], ABC):
    """
    Base class for all tasks. All subclasses of `Task` should implement the `execute` method.
    In case you are using some LLM (or DB for retrieval) which is still not supported by this library,
    you can write your own implementation easily by subclassing `Task`.

    :param name: The name of the task. The task name should be non-empty.
    :type name: str
    :raises ValueError: If task name fails validation.
    """

    def __init__(self, name: str) -> None:
        if not name or not name.strip():
            raise ValueError("A task should have a non-empty value for name")

        self._name = name

    @abstractmethod
    async def execute(self, task_input: TaskInput[T]) -> TaskOutput[U]:
        """
        Abstract method for executing the task.

        :param task_input: The input to the task.
        :type task_input: TaskInput[T]
        :returns: The output of the task.
        :rtype: TaskOutput[U]
        """
        pass

    def name(self) -> str:
        """
        Returns the name of the task.

        :returns: task name.
        :rtype: str
        """
        return self._name

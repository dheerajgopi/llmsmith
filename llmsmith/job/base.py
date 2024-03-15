from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Generic, List, TypeVar, Union

from llmsmith.task.base import Task
from llmsmith.task.models import TaskInput, TaskOutput


@dataclass
class JobMemory:
    """
    JobMemory is a key-value store which stores the input and output values for the tasks (:class:`llmsmith.task.base.Task`) present in the job.
    """

    def __init__(self) -> None:
        self.inputs: dict[str, TaskInput] = {}
        self.outputs: dict[str, TaskOutput] = {}

    def add_task_input(self, key: str, task_input: TaskInput):
        """
        Stores the input value passed to a task against the specified key.

        :param key: key against which input value is stored
        :type key: str
        :param task_input: input of the task
        :type task_input: :class:`llmsmith.task.models.TaskInput`
        """
        self.inputs[key] = task_input

    def add_task_output(self, key: str, task_output: TaskOutput):
        """
        Stores the output value returned by a task against the specified key.

        :param key: key against which output value is stored
        :type key: str
        :param task_output: output of the task
        :type task_output: :class:`llmsmith.task.models.TaskOutput`
        :raises KeyError: if there is no input value stored against the task
        """
        self.outputs[key] = task_output

    def get_task_input(self, key: str) -> Union[TaskInput, None]:
        """
        Returns the task input from the memory.

        :param key: key against which input value is stored
        :type key: str
        :return: input value passed to the task. Returns `None` if the key is not available in the memory
        :rtype: :class:`llmsmith.task.models.TaskInput`
        """
        if key not in self.inputs:
            return None
        return self.inputs[key]

    def get_task_output(self, key: str) -> Union[TaskOutput, None]:
        """
        Returns the task output from the memory.

        :param key: key against which output value is stored
        :type key: str
        :return: output value returned by the task. Returns `None` if the key is not available in the memory
        :rtype: :class:`llmsmith.task.models.TaskOutput`
        """
        if key not in self.outputs:
            return None
        return self.outputs[key]


class _JobTask:
    """
    `_JobTask` is a wrapper on top of :class:`llmsmith.task.base.Task` which is used internally by :class:`llmsmith.job.base.Job`.
    It executes a task as part of an ongoing job and stores the input and output values of the task in the :class:`llmsmith.task.base.JobMemory` instance.

    :param task: The task to be executed.
    :type task: :class:`llmsmith.task.base.Task`
    :param input_template: input template with placeholders referring to inputs/outputs of previous tasks.
    :type input_template: str
    """

    def __init__(self, task: Task, input_template: str) -> None:
        self.task = task
        self.input_template = input_template

    async def execute(self, user_input: str, memory: JobMemory):
        """
        Executes the task and stores the input and output values in the job memory.
        Before executing the task, the placeholders in the input template will be replaced
        with the corresponding values picked from the job memory.

        :param user_input: Initial user input.
        :type user_input: str
        :param memory: The job memory to store task input and output.
        :type memory: :class:`llmsmith.task.base.JobMemory`
        """
        input_updated_with_placeholders = self._replace_input_template_placeholders(
            user_input, memory
        )

        task_input = TaskInput(content=input_updated_with_placeholders)

        memory.add_task_input(key=self.task.name(), task_input=task_input)
        task_output = await self.task.execute(task_input)
        memory.add_task_output(key=self.task.name(), task_output=task_output)

    def task_name(self) -> str:
        """
        Returns the name of the task.

        :return: The name of the task.
        :rtype: str
        """
        return self.task.name()

    def _replace_input_template_placeholders(
        self, user_input: str, memory: JobMemory
    ) -> str:
        """
        Utility method for replacing the placeholders in the input template with the
        corresponding values picked from the job memory.
        """
        updated_input = self.input_template.replace("{{root}}", user_input)

        for key, inp in memory.inputs.items():
            updated_input = updated_input.replace(f"{{{{{key}.input}}}}", inp.content)

        for key, out in memory.outputs.items():
            updated_input = updated_input.replace(f"{{{{{key}.output}}}}", out.content)

        return updated_input


T = TypeVar("T")
U = TypeVar("U")


class Job(Generic[T, U], ABC):
    """
    A subclass of `Job` can run a list of tasks and store their inputs and outputs.
    An instance of :class:`llmsmith.task.base.JobMemory` is used internally for storing the task inputs/outputs.
    """

    def __init__(self) -> None:
        self._tasks: List[_JobTask] = []
        self._memory: JobMemory = JobMemory()

    @abstractmethod
    async def run(self, user_input: T):
        """
        Abstract method to run the Job with the given initial user input.

        :param user_input: The input provided by the user.
        :type user_input: T
        """
        pass

    def task_input(self, key: str) -> Union[TaskInput, None]:
        """
        Return the input for a task.

        :param key: The key of the task.
        :type key: str
        :return: The input of the task or None if not found.
        :rtype: Union[TaskInput, None]
        """
        return self._memory.get_task_input(key)

    def task_output(self, key: str) -> Union[TaskOutput, None]:
        """
        Return the output of a task.

        :param key: The key of the task.
        :type key: str
        :return: The output of the task or None if not found.
        :rtype: Union[TaskOutput, None]
        """
        return self._memory.get_task_output(key)

    @classmethod
    def _validate_task_names(cls, tasks: List[_JobTask]):
        """
        Checks for tasks with duplicate names in the job.

        :param tasks: The list of tasks.
        :type tasks: List[_JobTask]
        :raises ValueError: If a duplicate task name is found.
        """
        task_name_counts = {}
        for task in tasks:
            task_name_counts[task.task_name()] = (
                task_name_counts.get(task.task_name(), 0) + 1
            )

            if task_name_counts[task.task_name()] > 1:
                raise ValueError(
                    f"Duplicate task name '{task.task_name()}' present in the job"
                )

    @classmethod
    def _validate_task_types(cls, tasks: List[Any]):
        """
        Checks if only subclasses of `llmsmith.task.base.Task` as passed as tasks for the job.

        :param tasks: The list of tasks.
        :type tasks: List[Any]
        :raises TypeError: If a task is not a subclass of `llmsmith.task.base.Task`.
        """
        for task in tasks:
            if not isinstance(task, Task):
                raise TypeError(
                    "Only subclasses of `llmsmith.task.base.Task` are allowed as tasks"
                )

import asyncio
from typing import TypeVar, Union
from typing_extensions import Self

from llmsmith.job.base import Job, _JobTask
from llmsmith.task.base import Task


T = TypeVar("T")
U = TypeVar("U")


class SequentialJob(Job):
    """
    An implementation of :class:`llmsmith.job.base.Job` which executes the given tasks sequentially.
    When adding a task, it is possible to pass the input/output values of previous tasks via an input template with placeholders.
    The placeholders can be in the following formats:

    * For replacing with the input value of a previous task: {{task-name.input}}
    * For replacing with the output value of a previous task: {{task-name.output}}
    * For replacing with the initial user input which is passed to the job while running it: {{root}}

    A simple RAG implementation can be used as an example here for showcasing the above points.

    Consider the below flow:

    * An user query is passed as input to a retriever (chromaDB)
    * Retriever output is passed to an LLM (OpenAI) to rephrase the query
    * Rephrased query is used as input to an LLM (OpenAI) to get the answer

    Following is the code for the above flow using llmsmith.

    .. code-block:: python

        import chromadb
        import openai

        from chromadb.utils import embedding_functions

        from llmsmith.job.job import SequentialJob

        from llmsmith.task.retrieval.vector.chromadb import ChromaDBRetriever
        from llmsmith.task.textgen.options.openai import OpenAITextGenOptions
        from llmsmith.task.textgen.openai import OpenAITextGenTask

        chroma_client = chromadb.HttpClient(host=CHROMA_DB_HOST, port=8000)
        llm = openai.AsyncOpenAI(api_key=OPEN_AI_API_KEY)
        collection = chroma_client.get_collection(
            name="university_faq", embedding_function=embedding_functions.ONNXMiniLM_L6_V2()
        )

        user_input = "I'm interested in ML and AI. Tell me about the courses offered here which will match my interests."

        retrieval_task = ChromaDBRetriever(
            name="chromadb-retriever",
            collection=collection,
        )

        rephrase_task = OpenAITextGenTask(
            name="openai-rephraser",
            llm=llm,
            llm_options=OpenAITextGenOptions(model="gpt-3.5-turbo", temperature=0),
        )

        generate_answer_task = OpenAITextGenTask(
            name="openai-answer-generator",
            llm=llm,
            llm_options=OpenAITextGenOptions(model="gpt-3.5-turbo", temperature=0),
        )

        job = SequentialJob()

        # First step - retrieve the relevant documents from Chroma DB
        job.add_task(retrieval_task)

        # Second step - Rephrase the question based on the documents retrieved from Chroma DB.
        # Note the placeholders {{root}} and {{chromadb-retriever.output}}.
        # {{chromadb-retriever.output}} will be replaced by the output value of Chroma DB retriever
        # {{root}} will be replaced with the initial user input
        job.add_task(
            rephrase_task,
            input_template="Rephrase the question based on the context: \\n\\n QUESTION:\\n{{root}}\\n\\nCONTEXT:\\n{{chromadb-retriever.output}}",
        )

        # Third step - Answer the question based on the rephrased question and the relevant context documents retrieved from Chroma DB.
        # Note the placeholders {{openai-rephraser.output}} and {{chromadb-retriever.output}}.
        # {{openai-rephraser.output}} will be replaced by the output value of query rephraser task
        # {{chromadb-retriever.output}} will be replaced by the output value of Chroma DB retriever
        job.add_task(
            generate_answer_task,
            input_template="Answer the question based on the context: \\n\\n QUESTION:\\n{{openai-rephraser.output}}\\n\\nCONTEXT:\\n{{chromadb-retriever.output}}",
        )

        # Run the job. The 3 steps will be executed sequentially
        await job.run(user_input)

        # Print the output of the final task in the job
        print(job.task_output("openai-answer-generator"))
    """

    def __init__(self) -> None:
        super().__init__()

    def add_task(
        self, task: Task, input_template: Union[str, None] = "{{root}}"
    ) -> Self:
        """
        Add a task to the job. An optional input template can also be passed which can be
        used to pass the input/output values of previous tasks via placeholders.

        :param task: task to be added to the job
        :type task: :class:`llmsmith.task.base.Task`
        :param input_template: string template with placeholders referring to inputs/outputs of previous tasks.
            Defaults to `{{root}}` which refers to initial user input.
        :type input_template: str, optional
        :returns: Self
        :rtype: :class:`llmsmith.job.job.SequentialJob`
        """
        chain_task = _JobTask(task=task, input_template=input_template)
        self._tasks.append(chain_task)
        self._validate_task_types([task])
        self._validate_task_names(self._tasks)

        return self

    async def run(self, user_input: T):
        """
        Run the tasks sequentially.

        :param user_input: The initial input for the job
        :type user_input: T
        """
        for task in self._tasks:
            await task.execute(user_input, self._memory)


class ConcurrentJob(Job):
    """
    An implementation of :class:`llmsmith.job.base.Job` which executes the given tasks concurrently.
    Every task added to an instance of `ConcurrentJob` will share the same input, which is the initial user input passed to the job.

    Consider the below tasks for example:

    * Write a crime thriller movie idea based on the information given by the user.
    * Write a horror movie idea based on the information given by the user.

    Since both tasks are based on the same input provided by the user, we can do it concurrently.

    Following is the code for the above flow using llmsmith.

    .. code-block:: python

        import openai

        from llmsmith.job.job import ConcurrentJob
        from llmsmith.task.textgen.options.openai import OpenAITextGenOptions
        from llmsmith.task.textgen.openai import OpenAITextGenTask

        llm = openai.AsyncOpenAI(api_key=OPEN_AI_API_KEY)

        user_input = "Write a movie idea based on the below information:\\n Protagonist: introvert college student\\n Country: Japan\\n Location: university\\n"

        crime_movie_task = OpenAITextGenTask(
            name="openai-crime-movie-idea",
            llm=llm,
            llm_options=OpenAITextGenOptions(
                model="gpt-3.5-turbo",
                temperature=0.3,
                system_prompt="You are a movie script writer working on a crime thriller movie script"
            ),
        )

        horror_movie_task = OpenAITextGenTask(
            name="openai-horror-movie-idea",
            llm=llm,
            llm_options=OpenAITextGenOptions(
                model="gpt-3.5-turbo",
                temperature=0.3,
                system_prompt="You are a movie script writer working on a horror movie script"
            ),
        )

        job = ConcurrentJob()

        # Add task for writing crime thriller movie idea
        job.add_task(crime_movie_task)

        # Add task for writing horror movie idea
        job.add_task(horror_movie_task)

        # Run the job. The 2 steps will be executed concurrently
        await job.run(user_input)

        # Print the output of the both tasks
        print(job.task_output("openai-crime-movie-idea"))
        print(job.task_output("openai-horror-movie-idea"))
    """

    def __init__(self) -> None:
        super().__init__()

    def add_task(self, task: Task) -> Self:
        """
        Add a task to the job.

        :param task: task to be added to the job
        :type task: :class:`llmsmith.task.base.Task`
        :returns: Self
        :rtype: :class:`llmsmith.job.job.ConcurrentJob`
        """
        chain_task = _JobTask(task=task, input_template="{{root}}")
        self._tasks.append(chain_task)
        self._validate_task_types([task])
        self._validate_task_names(self._tasks)

        return self

    async def run(self, user_input: T):
        """
        Run the tasks concurrently.

        :param user_input: The initial input for the job
        :type user_input: T
        """
        await asyncio.gather(
            *[task.execute(user_input, self._memory) for task in self._tasks]
        )

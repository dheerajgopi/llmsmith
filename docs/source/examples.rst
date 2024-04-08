.. _examples-label:

Examples
========

A naive RAG function
--------------------

Let's create a naive RAG function using ``LLMSmith``. For this example, let's assume that the embeddings are stored in `Chroma DB <https://www.trychroma.com/>`_, and ``gpt-3.5-turbo`` from `OpenAI <https://openai.com/>`_ is used as the LLM model.

For this example, we are going to create a simple python function which performs RAG using ``LLMSmith``.

Firstly, add ``LLMSmith`` to your Python project with the following command.

.. code-block:: console

    pip install "llmsmith[openai,chromadb]"

In case you are using poetry, use the following command instead.

.. code-block:: console

    poetry add 'llmsmith[openai,chromadb]'

and now, lets check the code for the RAG function.

.. code-block:: python

    import asyncio
    import os

    import chromadb
    from chromadb.utils import embedding_functions
    import openai
    from llmsmith.job.job import SequentialJob

    from llmsmith.task.retrieval.vector.chromadb import ChromaDBRetriever
    from llmsmith.task.textgen.options.openai import OpenAITextGenOptions
    from llmsmith.task.textgen.openai import OpenAITextGenTask

    async def run_rag(user_prompt):
        # Creaet ChromaDB client
        chroma_client = chromadb.HttpClient(host="localhost", port=8000)

        # Create async OpenAI client
        llm = openai.AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        # if you are using local LLM using Ollama, use the below line instead
        # llm = openai.AsyncOpenAI(api_key="sk-api-key", base_url="http://localhost:11434/v1/")

        # Create a client for the Chroma DB collection (`test_collection`)
        collection: chromadb.Collection = chroma_client.get_collection(
            name="test_collection", embedding_function=embedding_functions.ONNXMiniLM_L6_V2()
        )

        # define the retriever task
        retrieval_task: ChromaDBRetriever = ChromaDBRetriever(
            name="chromadb-retriever",
            collection=collection,
        )

        # define the LLM task for answering the query
        generate_answer_task: OpenAITextGenTask = OpenAITextGenTask(
            name="openai-answer-generator",
            llm=llm,
            llm_options=OpenAITextGenOptions(model="gpt-3.5-turbo", temperature=0),
        )

        # define the sequence of tasks
        # {{root}} is a special placeholer in `input_template` which will be replaced with the prompt entered by the user (`user_prompt`)
        # the placeholder {{chromadb-retriever.output}} will be replaced with the output from Chroma DB retriever task.
        job: SequentialJob[str, str] = (
            SequentialJob()
            .add_task(retrieval_task)
            .add_task(
                generate_answer_task,
                input_template="Answer the question based on the context: \n\n QUESTION:\n{{root}}\n\nCONTEXT:\n{{chromadb-retriever.output}}",
            )
        )

        # Now, run the job
        await job.run(user_prompt)

        # return the output
        return job.task_output("openai-answer-generator")
    
    if __name__ == "__main__":
        loop = asyncio.get_event_loop()
        loop.run_until_complete(run_rag("your query goes here"))

Now, its just a matter of calling ``await run_rag("your query goes here")`` wherever you need the RAG functionality.

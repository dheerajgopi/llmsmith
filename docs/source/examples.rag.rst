Retrieval Augmented Generation (RAG)
====================================

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
    import logging
    import os
    import sys

    import chromadb
    from chromadb.utils import embedding_functions
    from dotenv import load_dotenv
    import openai
    from llmsmith.job.job import SequentialJob

    from llmsmith.task.retrieval.vector.chromadb import ChromaDBRetriever
    from llmsmith.task.textgen.options.openai import OpenAITextGenOptions
    from llmsmith.task.textgen.openai import OpenAITextGenTask

    load_dotenv()

    log_handler = logging.StreamHandler(sys.stdout)
    log = logging.getLogger(__name__)
    log.addHandler(log_handler)
    log.setLevel(logging.INFO)

    logging.getLogger("llmsmith").addHandler(log_handler)
    logging.getLogger("llmsmith").setLevel(logging.DEBUG)


    async def run_rag(user_prompt):
        # Create ChromaDB client
        chroma_client = chromadb.HttpClient(host="localhost", port=8000)

        # Create async OpenAI client
        llm = openai.AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        # if you are using local LLM using Ollama, use the below line instead
        # llm = openai.AsyncOpenAI(api_key="sk-api-key", base_url="http://localhost:11434/v1/")

        # Create a client for the Chroma DB collection (`test_collection`)
        collection: chromadb.Collection = chroma_client.get_collection(
            name="test_collection"
        )

        # define the retriever task along with the embedding function
        retrieval_task: ChromaDBRetriever = ChromaDBRetriever(
            name="chromadb-retriever",
            collection=collection,
            embedding_func=lambda x: embedding_functions.ONNXMiniLM_L6_V2().embed_with_retries(
                x
            ),
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

        log.info(job.task_output("openai-answer-generator").content)

        # return the output
        return job.task_output("openai-answer-generator")


    if __name__ == "__main__":
        loop = asyncio.get_event_loop()
        loop.run_until_complete(
            run_rag("what sort of clubs are available in the university?")
        )


Now, its just a matter of calling ``await run_rag("your query goes here")`` wherever you need the RAG functionality.


Advanced RAG with pre-processing and reranking
----------------------------------------------

A naive RAG (simple vector DB + LLM combo) is easy to implement. But most of the time, the results leave a lot to be desired.
One of the easiest and quickest way to increase the quality of results produced by the RAG system is to add a reranker into the mix.
A reranker will calculate similarity score based on the query and document pair, and use this score to reorder the documents retrieved from vector DB by relevance to the query.

Another optimization we can do is to pre-process the user's question (using an LLM) before retrieving documents from the vector database.
The pre-processing step can be used to remove information from the user's query which are irrelevant for the retrieval task.
This can improve the quality of documents retrieved from the vector database.

Incorporating the above mentioned optimizations, the RAG flow will be as given below.

.. code-block:: console

    user query -> pre-process user's query -> retrieve documents -> rerank documents -> answer the user's query

Let's implement the above flow using ``LLMSmith``. We will be using

* ``gemini-pro`` from `Google Gemini <https://gemini.google.com>`_ for query pre-processing.
* `Qdrant <https://qdrant.tech>`_ as vector database.
* `Cohere <https://docs.cohere.com/docs/rerank-2>`_ for reranking.
* ``gpt-4-turbo`` from `OpenAI <https://openai.com/>`_ for generating answer based on reranked documents.

Firstly, add ``LLMSmith`` to your Python project with the following command.

.. code-block:: console

    pip install "llmsmith[openai,gemini,qdrant,cohere]"

In case you are using poetry, use the following command instead.

.. code-block:: console

    poetry add 'llmsmith[openai,gemini,qdrant,cohere]'

For this example, we need to install ``fastembed`` too, since that is used for embedding documents.

.. code-block:: console

    pip install fastembed

or

.. code-block:: console

    poetry add fastembed

and now, lets check the code.

.. code-block:: python

    import asyncio
    import logging
    import os
    import sys
    from textwrap import dedent

    import cohere
    from dotenv import load_dotenv
    import google.generativeai as genai
    from google.generativeai.types import GenerationConfig
    import openai
    from qdrant_client import AsyncQdrantClient

    from fastembed import TextEmbedding

    from llmsmith.job.job import SequentialJob
    from llmsmith.reranker.cohere import CohereReranker

    from llmsmith.task.retrieval.vector.qdrant import QdrantRetriever
    from llmsmith.task.textgen.gemini import GeminiTextGenTask
    from llmsmith.task.textgen.openai import OpenAITextGenTask
    from llmsmith.task.textgen.options.gemini import GeminiTextGenOptions
    from llmsmith.task.textgen.options.openai import OpenAITextGenOptions


    load_dotenv()

    log_handler = logging.StreamHandler(sys.stdout)
    log = logging.getLogger(__name__)
    log.addHandler(log_handler)
    log.setLevel(logging.INFO)

    logging.getLogger("llmsmith").addHandler(log_handler)
    logging.getLogger("llmsmith").setLevel(logging.DEBUG)


    async def run_rag(user_prompt: str):
        # Create Gemini client
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        gemini_llm = genai.GenerativeModel("gemini-pro")

        # Create OpenAI client
        openai_llm = openai.AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        # Create Cohere client
        cohere_client = cohere.AsyncClient(api_key=os.getenv("COHERE_API_KEY"))

        # Create Qdrant client
        qdrant_client = AsyncQdrantClient(host="localhost", port=6333)

        # For this example, assume fastembed is used for embedding the documents inserted into Qdrant.
        embed = TextEmbedding("BAAI/bge-small-en")

        # Create Cohere reranker
        reranker = CohereReranker(client=cohere_client)

        # Define the Qdrant retriever task. The embedding function and reranker are passed as parameters.
        retrieval_task = QdrantRetriever(
            name="qdrant-retriever",
            client=qdrant_client,
            collection_name="test",
            embedding_func=lambda x: list(embed.query_embed(x)),
            embedded_field_name="description",  # name of the field in the document on which embeddedings are created while uploading data to the Qdrant collection
            reranker=reranker,
        )

        # Define the Gemini LLM task for rephrasing the query
        preprocess_task = GeminiTextGenTask(
            name="gemini-preprocessor",
            llm=gemini_llm,
            llm_options=GeminiTextGenOptions(
                generation_config=GenerationConfig(temperature=0)
            ),
        )

        # Define the OpenAI LLM task for answering the query
        answer_generate_task = OpenAITextGenTask(
            name="openai-answer-generator",
            llm=openai_llm,
            llm_options=OpenAITextGenOptions(model="gpt-4-turbo", temperature=0),
        )

        # define the sequence of tasks
        # {{root}} is a special placeholer in `input_template` which will be replaced with the prompt entered by the user (`user_prompt`).
        # The placeholder {{qdrant-retriever.output}} will be replaced with the output from Qdrant DB retriever task.
        # The placeholder {{gemini-preprocessor.output}} will be replaced with the output from the query preprocessing task done by Gemini LLM.
        job: SequentialJob[str, str] = (
            SequentialJob()
            .add_task(
                preprocess_task,
                input_template=dedent("""
                    Convert the natural language query from a user into a query for a vectorstore.
                    In this process, you strip out information that is not relevant for the retrieval task.
                    Here is the user query: {{root}}""")
                .strip("\n")
                .replace("\n", " "),
            )
            .add_task(retrieval_task, input_template="{{gemini-preprocessor.output}}")
            .add_task(
                answer_generate_task,
                input_template="Answer the question based on the context: \n\n QUESTION:\n{{root}}\n\nCONTEXT:\n{{qdrant-retriever.output}}",
            )
        )

        # Now, run the job
        await job.run(user_prompt)

        log.info(job.task_output("openai-answer-generator").content)

        # return the output
        return job.task_output("openai-answer-generator")


    if __name__ == "__main__":
        loop = asyncio.get_event_loop()
        loop.run_until_complete(run_rag("what sort of clubs are available in the university?"))

Now, its just a matter of calling ``await run_rag("your query goes here")`` wherever you need the RAG functionality.

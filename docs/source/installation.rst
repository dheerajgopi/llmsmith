Installation
============

``LLMSmith`` does not download extra dependencies (like openai, google-generativeai, chromadb etc.) by default. This is to minimize bloat and keep your application package size in check. The recommended way to install LLMSmith is by specifying the extra dependencies explicitly.

For example, if your project is using an OpenAI LLM and Chroma DB vector database, install LLMSmith using the below command.

.. code-block:: console

    pip install "llmsmith[openai,chromadb]"

The above command ensures that only the required dependencies (openai and chromadb clients in this case) are downloaded. The rest are ignored, thus reducing the package size.

Here's the list of extra dependencies supported by LLMSmith:

- `openai`

- `claude`

- `gemini`

- `chromadb`

- `all` (downloads all extra dependencies)
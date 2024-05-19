# 🧰 LLMSmith

## What is LLMSmith?

**LLMSmith** is a lightweight Python library designed for developing functionalities powered by Large Language Models (LLMs). It allows developers to integrate LLMs into various types of applications, whether they are web applications, GUI applications, or any other kind of application.

## Installation

LLMSmith does not download extra dependencies (like openai, google-generativeai, chromadb etc.) by default. This is to minimize bloat and keep your application package size in check. The recommended way to install LLMSmith is by specifying the extra dependencies explicitly.

For example, if your project is using an OpenAI LLM and Chroma DB vector database, install LLMSmith using the below command.

```
pip install "llmsmith[openai,chromadb]"
```

The above command ensures that only the required dependencies (openai and chromadb clients in this case) are downloaded. The rest are ignored, thus reducing the package size.

Here's the list of extra dependencies supported by LLMSmith:
- `openai`
- `claude`
- `gemini`
- `chromadb`
- `qdrant`
- `cohere`
- `pinecone`
- `all` (downloads all extra dependencies)

## Example

Refer [this](https://llmsmith.readthedocs.io/en/latest/examples.html) page from the documentation to see examples on how LLMSmith can be used to build LLM powered functionalities.

## Documentation

An extensive documentation for LLMSmith is hosted in Read the Docs here - [https://llmsmith.readthedocs.io/en/latest](https://llmsmith.readthedocs.io/en/latest)

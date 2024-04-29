Function Calling Agents
=======================

``LLMSmith`` comes with an abstraction for agent loops based on function calling capabilities of LLMs (supports OpenAI and Gemini as of now).
So, instead of writing the agent loops manually, simply create an instance of LLMSmith function calling agent by passing the LLM client along with its configuration and LLM specific tool declarations.
The LLMSmith function calling agents can also be added as a ``task`` to an LLMSmith ``job``.

OpenAI Function Calling Agent
-----------------------------

.. code-block:: console

    pip install "llmsmith[openai]"

In case you are using poetry, use the following command instead.

.. code-block:: console

    poetry add 'llmsmith[openai]'

and now, lets check the code which utilizes the OpenAI function calling agent.

.. code-block:: python

    import asyncio
    import logging
    import os
    import sys

    from dotenv import load_dotenv
    import openai
    from llmsmith.agent.function.openai import OpenAIFunctionAgent
    from llmsmith.agent.function.options.openai import OpenAIAssistantOptions
    from llmsmith.agent.tool.openai import OpenAIAssistantTool

    from llmsmith.task.models import TaskInput


    # load env vars for getting OPENAI_API_KEY
    load_dotenv()

    # Enable debug logs for agent to view the responses in agent loop
    log_handler = logging.StreamHandler(sys.stdout)
    logging.getLogger("llmsmith.agent").addHandler(log_handler)
    logging.getLogger("llmsmith.agent").setLevel(logging.DEBUG)


    # Define the functions which will be the part of the LLM toolkit
    def add(a: float, b: float) -> float:
        return a + b


    async def run():
        # initialize OpenAI client
        llm = openai.AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        # declare the tools which can be used by OpenAI LLM
        add_tool = OpenAIAssistantTool(
            declaration={
                "function": {
                    "name": "add",
                    "description": "Returns the sum of two numbers.",
                    "parameters": {
                        "type": "object",
                        "properties": {"a": {"type": "number"}, "b": {"type": "number"}},
                        "required": ["a", "b"],
                    },
                },
                "type": "function",
            },
            callable=add,
        )

        # create the agent
        task: OpenAIFunctionAgent = await OpenAIFunctionAgent.create(
            name="testfunc",
            llm=llm,
            assistant_options=OpenAIAssistantOptions(model="gpt-4-turbo"),
            tools=[add_tool],
            max_turns=5,
        )

        # run the agent
        res = await task.execute(TaskInput("Add sum of 1 and 2 to the sum of 5 and 6"))

        print(f"\n\nAgent response: {res.content}")


    if __name__ == "__main__":
        loop = asyncio.get_event_loop()
        loop.run_until_complete(run())


Gemini Function Calling Agent
-----------------------------

.. code-block:: console

    pip install "llmsmith[gemini]"

In case you are using poetry, use the following command instead.

.. code-block:: console

    poetry add 'llmsmith[gemini]'

and now, lets check the code which utilizes the Gemini function calling agent.

.. code-block:: python

    import asyncio
    import logging
    import os
    import sys

    from dotenv import load_dotenv
    import google.generativeai as genai
    from google.generativeai.types import GenerationConfig

    from llmsmith.agent.function.gemini import GeminiFunctionAgent
    from llmsmith.agent.tool.gemini import GeminiTool
    from llmsmith.task.models import TaskInput

    from llmsmith.task.textgen.options.gemini import GeminiTextGenOptions


    # load env vars for getting GOOGLE_API_KEY
    load_dotenv()

    # Enable debug logs for agent to view the responses in agent loop
    log_handler = logging.StreamHandler(sys.stdout)
    logging.getLogger("llmsmith.agent").addHandler(log_handler)
    logging.getLogger("llmsmith.agent").setLevel(logging.DEBUG)


    # Define the functions which will be part of the LLM toolkit
    def multiply(a: float, b: float) -> float:
        return a * b


    def add(a: float, b: float) -> float:
        return a + b


    def subtract(a: float, b: float) -> float:
        return a - b


    async def run():
        # initialize Gemini client
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        llm = genai.GenerativeModel("gemini-pro")

        # declare the tools (functions) which can be used by Gemini LLM
        calculator_tools = [
            GeminiTool(
                declaration={
                    "name": "add",
                    "description": "Returns the sum of two numbers.",
                    "parameters": {
                        "type_": "OBJECT",
                        "properties": {"a": {"type_": "NUMBER"}, "b": {"type_": "NUMBER"}},
                        "required": ["a", "b"],
                    },
                },
                callable=add,
            ),
            GeminiTool(
                declaration={
                    "name": "multiply",
                    "description": "Returns the product of two numbers.",
                    "parameters": {
                        "type_": "OBJECT",
                        "properties": {"a": {"type_": "NUMBER"}, "b": {"type_": "NUMBER"}},
                        "required": ["a", "b"],
                    },
                },
                callable=multiply,
            ),
            GeminiTool(
                declaration={
                    "name": "subtract",
                    "description": "Returns the difference of two numbers.",
                    "parameters": {
                        "type_": "OBJECT",
                        "properties": {"a": {"type_": "NUMBER"}, "b": {"type_": "NUMBER"}},
                        "required": ["a", "b"],
                    },
                },
                callable=subtract,
            ),
        ]

        # create the agent
        agent: GeminiFunctionAgent = GeminiFunctionAgent(
            name="func_call",
            llm=llm,
            llm_options=GeminiTextGenOptions(
                generation_config=GenerationConfig(temperature=0),
            ),
            tools=calculator_tools,
            max_turns=5,
        )

        # run the agent
        res = await agent.execute(
            TaskInput("calculate sum of 1 and 5 and multiply it with difference of 6 and 3")
        )
        print(f"\n\nAgent response: {res.content}")


    if __name__ == "__main__":
        loop = asyncio.get_event_loop()
        loop.run_until_complete(run())
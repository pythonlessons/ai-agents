import os
import json
from typing import List, Optional, Union, Type

from openai import OpenAI
from dotenv import load_dotenv
from pydantic import BaseModel
from openai.types.chat import ChatCompletionMessage
load_dotenv()

from tools import RetrievalTool, AnalyzerTool, InferTableSchemaTool, BuildDataFrameTool, SaveDataFrameTool

class Agent:
    """
    An LLM-powered agent that can use multiple tools to solve tasks.
    
    This agent manages conversations with an LLM, allows the model to decide
    when to use registered tools, and handles the execution of those tools
    in a sequential manner until the task is complete.
    
    Attributes:
        model (str): The name of the LLM model to use.
        client (OpenAI): The OpenAI client instance for making API calls.
        tools (Dict[str, callable]): Dictionary mapping tool names to tool instances.
        tool_definitions (List[Dict]): List of tool definitions for the OpenAI API.
        messages (List): The conversation history.
        debug (bool): Whether to print debug information.
        reset_before_tool (object): A tool class; if the model decides to call this tool,
            the agent will reset the conversation history before calling it. Defaults to None.
    """

    def __init__(
        self,
        client: OpenAI,
        system_prompt: str = "You are a helpful assistant.",
        tools: List = [],
        model: str = "gpt-4.1",
        debug: bool = False,
        reset_before_tool: object = None,
    ):
        """
        Initialize the Agent with a client, system prompt, and tools.
        
        Args:
            client (OpenAI): The OpenAI client instance for making API calls.
            system_prompt (str, optional): The system prompt that defines the agent's behavior.
                Defaults to "You are a helpful assistant.".
            tools (List, optional): List of tool instances to register with the agent.
                Each tool should have a name attribute and be callable. Defaults to [].
            model (str, optional): The name of the LLM model to use. Defaults to "gpt-4.1".
            debug (bool, optional): Whether to print debug information. Defaults to False.
            reset_before_tool (object, optional): A tool class; if the model decides to call this tool,
                the agent will reset the conversation history before calling it. Defaults to None.
        """
        self.model = model
        self.client = client
        self.tools = {tool.name: tool for tool in tools}
        self.tool_definitions = [tool.tool() for tool in tools]
        self.system_prompt = system_prompt
        self.reset_before_tool = reset_before_tool
        self.messages: List[Union[dict, ChatCompletionMessage]] = [
            {"role": "system", "content": system_prompt}
        ]
        self.debug = debug

    def _log_debug(self, *args) -> None:
        """
        Log debug information if debug mode is enabled.
        
        Args:
            *args: Values to log.
        """
        if self.debug:
            print("[DEBUG]", *args)

    def _reset_session(self, user_input: str, extra_messages: List[dict]=None) -> None:
        """Clear conversation history and restart with the same user input.
        """
        self._log_debug("Resetting session.")
        self.messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_input},
        ] 
        if extra_messages:
            self.messages.extend(extra_messages)

    def get_tool_by_name(self, name: str):
        if name not in self.tools:
            raise ValueError(f"Tool '{name}' not registered.")
        return self.tools[name]

    def run(
        self,
        user_input: str,
        response_format: Optional[Type[BaseModel]] = None,
    ) -> Union[str, BaseModel]:
        """
        Run the agent with a user input until it reaches a conclusion.
        
        This method takes a user input, sends it to the LLM, and then handles
        any tool calls the LLM wants to make. It continues this process until
        the LLM provides a final response without requesting any tool calls.
        
        Args:
            user_input (str): The user's input message.
            response_format (Optional[Type[BaseModel]], optional): A Pydantic model class
                to parse the final response into. Defaults to None.
                
        Returns:
            Union[str, BaseModel]: Either a string containing the LLM's final response,
                or an instance of the provided Pydantic model if response_format is specified.
        """
        self.messages.append({"role": "user", "content": user_input})
        self._log_debug("User input added to messages:", user_input)

        while True:
            self._log_debug("Sending messages to model...")
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=self.messages,
                tools=self.tool_definitions,
            )
            msg = completion.choices[0].message
            self._log_debug("Model response:", msg)

            tool_calls = msg.tool_calls

            if not tool_calls:
                self.messages.append(msg.model_dump())  # Final response
                self._log_debug("Final message from model:", msg.content)
                break

            self.messages.append(msg.model_dump())  # Tool proposal
            self._log_debug("Tool calls proposed:", tool_calls)

            for tool_call in tool_calls:
                name = tool_call.function.name
                args = json.loads(tool_call.function.arguments) if tool_call.function.arguments else {}
                print(f"Starting {name} tool")

                tool = self.get_tool_by_name(name)
                if isinstance(tool, self.reset_before_tool) and self.messages:
                    self._reset_session(user_input, extra_messages=[self.messages[-1]])

                self._log_debug(f"Calling tool: {name} with args: {args}")
                result = tool(**args)
                self._log_debug(f"Tool result: {result}")

                tool_msg = {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(result),
                }
                self.messages.append(tool_msg)
                self._log_debug("Tool message added to messages:", tool_msg)

        if response_format:
            self._log_debug("Parsing final response with format:", response_format)
            try:
                completion_parse = self.client.beta.chat.completions.parse(
                    model=self.model,
                    messages=self.messages,
                    tools=self.tool_definitions,
                    response_format=response_format,
                )
                parsed = completion_parse.choices[0].message.parsed
                self._log_debug("Parsed response:", parsed)
                return parsed
            except Exception as e:
                self._log_debug(f"Error parsing response: {e}")
                # If parsing fails, return the raw message content as fallback
                return msg.content

        return msg.content
    

if __name__ == "__main__":
    system_prompt = (
        "You are a helpful receipts analyst. Tool-use policy: "
        "1) Always call retrieve_documents after starting the session, with 'question' equal to the user's latest message, to gather context from the Vectorize pipeline. "
        "2) Then call analyze_documents with 'query' set to the same user question and 'context' set to the array returned by retrieve_documents. "
        "3) After you produce an analysis, ASK the user: 'Do you want this printed as-is or converted into a table?'. If table is requested, call infer_table_schema with the analysis text, then call build_dataframe with that analysis text and the returned columns. In your assistant message, do not DISPLAY the resulting table ass it is printed in python terminal"
        "4) After showing the table, ASK whether to save it to disk. If yes, call save_dataframe with the same columns and rows (and a user-provided path/format if given), then report the saved path. "
        "5) Only produce a final answer after analyzing with context. If there is no relevant context, say 'Sorry I don't know'."
    )

    vectorize_io_token = os.getenv("vectorize_io_token")
    organization_id = os.getenv("organization_id")
    pipeline_id = os.getenv("pipeline_id")
    base_url = f"https://api.vectorize.io/v1/org/{organization_id}/pipelines/{pipeline_id}"

    # General LLM creds for schema/df tools (use same as agent LLM)
    general_llm_token = os.getenv("OPENAI_API_KEY")
    general_llm_base = os.getenv("OPENAI_API_BASE_URL", "https://api.openai.com/v1")

    # Initialize OpenAI client (LLM)
    client = OpenAI(api_key=general_llm_token, base_url=general_llm_base)
    
    # Initialize agent with tools
    model="gpt-4.1"
    agent = Agent(
        client=client,
        system_prompt=system_prompt,
        tools=[
            RetrievalTool(vectorize_io_token, organization_id, pipeline_id, return_all_data=False),
            AnalyzerTool(general_llm_token, general_llm_base, model),
            InferTableSchemaTool(general_llm_token, general_llm_base, model),
            BuildDataFrameTool(general_llm_token, general_llm_base, model),
            SaveDataFrameTool(),
            ],
        debug=False,
        model=model,
        reset_before_tool=RetrievalTool
    )

    while True:
        user_input = input("\n👤 You: ")
        if user_input.lower() in ["exit", "quit"]:
            break

        # Run the agent with the user input
        # Summarize my spending by category over all my purchases in 2014 ordering fron highest to lowest spent per category
        # Print my Walmart purchases with prices where I spent less than 10$ for each product in 2019
        # Summarize all my spendings from receipts by category, date of purchase, what I bought, where I bought it and how much it cost
        # Can you flag and list any unusual transactions from my purchase history?
        response = agent.run(user_input)
        print(f"\n🤖 Agent: {response}")
import os
import json
from typing import List, Optional, Union, Type

from openai import OpenAI
from dotenv import load_dotenv
from pydantic import BaseModel
from openai.types.chat import ChatCompletionMessage
load_dotenv()

from tools import EmailSupportTool, AmazonPolicyTool, ConfirmTool, ImproveMessageTool

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
    """

    def __init__(
        self,
        client: OpenAI,
        system_prompt: str = "You are a helpful assistant.",
        tools: List = [],
        model: str = "gpt-4.1",
        debug: bool = False,
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
        """
        self.model = model
        self.client = client
        self.tools = {tool.name: tool for tool in tools}
        self.tool_definitions = [tool.tool() for tool in tools]
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

    def call_tool_by_name(self, name: str, args: dict) -> dict:
        """
        Call a registered tool by its name with the given arguments.
        
        Args:
            name (str): The name of the tool to call.
            args (dict): Arguments to pass to the tool.
            
        Returns:
            dict: The result of calling the tool.
            
        Raises:
            ValueError: If the tool name is not registered.
        """
        if name not in self.tools:
            raise ValueError(f"Tool '{name}' not registered.")
        self._log_debug(f"Calling tool: {name} with args: {args}")
        result = self.tools[name](**args)
        self._log_debug(f"Tool result: {result}")
        return result

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
                args = json.loads(tool_call.function.arguments)
                result = self.call_tool_by_name(name, args)

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
    system_prompt = "You are a helpful assistant that can answer questions, answer Amazon-related FAQs using internal documents, and email Amazon support if necessary. "
    
    # Initialize OpenAI client
    client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY"), 
        base_url=os.getenv("OPENAI_API_BASE_URL", "https://api.openai.com/v1")
    )
    
    # Initialize agent with tools
    agent = Agent(
        client=client,
        system_prompt=system_prompt,
        tools=[
            EmailSupportTool(), 
            AmazonPolicyTool(),
            ConfirmTool(),
            ImproveMessageTool(),
            ],
        debug=False,
        model=os.getenv("MODEL_NAME", "gpt-4.1"),
    )

    while True:
        user_input = input("\n👤 You: ")
        if user_input.lower() in ["exit", "quit"]:
            break

        response = agent.run(user_input)
        print(f"\n🤖 Agent: {response}")
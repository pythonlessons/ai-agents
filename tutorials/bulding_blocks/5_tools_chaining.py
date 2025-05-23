import os
import json
import math
import requests
from typing import List, Optional, Union, Type, Dict, Any

from openai import OpenAI
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from openai.types.chat import ChatCompletionMessage
load_dotenv()


class WeatherTool:
    """
    A tool for retrieving current weather conditions using Open-Meteo API.
    
    This tool allows querying current weather data including temperature
    and wind speed for a specific geographic location.
    
    Attributes:
        name (str): The name of the tool, used when registering with the agent.
    """

    def __init__(self, name: str = "get_weather"):
        """
        Initialize the WeatherTool with a name.
        
        Args:
            name (str, optional): The name to use for this tool. Defaults to "get_weather".
        """
        self.name = name

    def __call__(self, latitude: float, longitude: float) -> Dict[str, Any]:
        """
        Fetch current weather data for the given coordinates.
        
        Makes a request to the Open-Meteo API and returns the current weather data.
        
        Args:
            latitude (float): The latitude coordinate.
            longitude (float): The longitude coordinate.
            
        Returns:
            Dict[str, Any]: Dictionary containing current weather data including:
                - temperature_2m: Current temperature in Celsius
                - wind_speed_10m: Current wind speed in km/h
                - time: The timestamp for the data
                
        Raises:
            requests.RequestException: If the API request fails
            ValueError: If the response doesn't contain expected data
            json.JSONDecodeError: If the response isn't valid JSON
        """
        try:
            response = requests.get(
                f"https://api.open-meteo.com/v1/forecast?latitude={latitude}&longitude={longitude}&current=temperature_2m,wind_speed_10m&hourly=temperature_2m,relative_humidity_2m,wind_speed_10m"
            )
            response.raise_for_status()  # Raise exception for HTTP errors
            data = response.json()
            
            if "current" not in data:
                raise ValueError("API response missing 'current' data")
                
            return data["current"]
        except requests.RequestException as e:
            return {"error": f"API request failed: {str(e)}"}
        except (json.JSONDecodeError, KeyError) as e:
            return {"error": f"Failed to parse weather data: {str(e)}"}

    def tool(self) -> Dict[str, Any]:
        """
        Get the tool definition for OpenAI function calling.
        
        Returns:
            Dict[str, Any]: A dictionary defining the tool for the OpenAI API.
        """
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": "Get current temperature in Celsius for given coordinates.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "latitude": {"type": "number"},
                        "longitude": {"type": "number"},
                    },
                    "required": ["latitude", "longitude"],
                    "additionalProperties": False,
                },
                "strict": True,
            },
        }


class CalculatorTool:
    """
    A tool for evaluating mathematical expressions safely.
    
    This tool allows evaluating basic mathematical expressions provided as strings,
    with safety precautions to prevent code execution vulnerabilities.
    
    Attributes:
        name (str): The name of the tool, used when registering with the agent.
    """

    def __init__(self, name: str = "calculate_expression"):
        """
        Initialize the CalculatorTool with a name.
        
        Args:
            name (str, optional): The name to use for this tool. Defaults to "calculate_expression".
        """
        self.name = name
        # Define allowed operations for safer evaluation
        self._allowed_names = {
            k: v for k, v in vars(math).items() if not k.startswith("__")
        } if "math" in globals() else {}
        self._allowed_names.update({
            "abs": abs, "float": float, "int": int,
            "max": max, "min": min, "pow": pow, "round": round, "sum": sum
        })

    def __call__(self, expression: str) -> Dict[str, Any]:
        """
        Evaluate a mathematical expression.
        
        Args:
            expression (str): The mathematical expression to evaluate, e.g. "2 + 3 * 4".
            
        Returns:
            Dict[str, Any]: Dictionary with the result of the evaluation:
                - result: The numeric result of the expression
                - or an error message if evaluation failed
                
        Note:
            This implementation uses Python's eval() with restrictions for safety.
            Only mathematical operations are allowed; no attribute access or imports.
        """
        try:
            # Basic validation - reject expressions with suspicious patterns
            if any(forbidden in expression for forbidden in 
                  ["__", "import", "eval", "exec", "getattr", "setattr", 
                   "os.", "sys.", "open", "file", "compile"]):
                return {"error": "Expression contains forbidden operations"}
                
            # Try to evaluate the expression using Python's eval
            # This is still not 100% safe but better than unrestricted eval
            result = eval(expression, {"__builtins__": {}}, self._allowed_names)
            return {"result": result}
        except Exception as e:
            return {"error": f"Error evaluating expression: {str(e)}"}

    def tool(self) -> Dict[str, Any]:
        """
        Get the tool definition for OpenAI function calling.
        
        Returns:
            Dict[str, Any]: A dictionary defining the tool for the OpenAI API.
        """
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": "Evaluate a math expression, e.g., '23.5 + 18.2'.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "expression": {"type": "string", "description": "A mathematical expression to evaluate"},
                    },
                    "required": ["expression"],
                    "additionalProperties": False,
                },
                "strict": True,
            },
        }


class FinalResponse(BaseModel):
    """
    Schema for the final structured response from the agent.
    
    This Pydantic model defines the expected format for the final response
    when using the agent with a structured output format.
    
    Attributes:
        result (float): The final numeric result of a calculation or measurement.
        response (str): A human-readable explanation of the result and how it was obtained.
    """
    result: float = Field(description="The final numeric result of the calculation or temperature.")
    response: str = Field(description="Human-readable explanation of the result.")


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
    system_prompt = "You are a helpful assistant who can get weather and do calculations."
    
    # Initialize OpenAI client
    client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY"), 
        base_url=os.getenv("OPENAI_API_BASE_URL", "https://api.openai.com/v1")
    )
    
    # Initialize agent with tools
    agent = Agent(
        client=client,
        system_prompt=system_prompt,
        tools=[WeatherTool(), CalculatorTool()],
        debug=False,
        model=os.getenv("MODEL_NAME", "gpt-4.1"),
    )
    
    # Example user input
    # user_input = "What is the current temperature in New York?"
    user_input = "What is the sum of current temperature in Paris, Berlin and New York?"
    
    # Get response from agent
    response = agent.run(user_input, response_format=FinalResponse)
    
    # Print the result and response
    print(f"Result: {response.result}")
    print(f"Response: {response.response}")
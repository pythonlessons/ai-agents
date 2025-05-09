import os
import json
import requests
from openai import OpenAI
from pydantic import BaseModel, Field
from dotenv import load_dotenv
load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

"""
docs: https://platform.openai.com/docs/guides/function-calling
"""

# Define the database base retrieval tool
def search_db(question: str):
    """
    Load the whole example database from the JSON file.
    (This is a mock function for demonstration purposes, we don't search)
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    db_path = os.path.join(current_dir, "db.json")
    with open(db_path, "r") as f:
        return json.load(f)


# Step 1: Call model with search_db tool defined
tools = [
    {
        "type": "function",
        "function": {
            "name": "search_db",
            "description": "Get the answer to the user's question from the knowledge base.",
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {"type": "string"},
                },
                "required": ["question"],
                "additionalProperties": False,
            },
            "strict": True,
        },
    }
]

# Define the system prompt, and simulate the user question
system_prompt = "You are a helpful assistant that answers questions from the knowledge base about our e-commerce store."
messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": "What is the return policy?"},
]

completion = client.chat.completions.create(
    model=os.getenv("MODEL_NAME", "gpt-4.1"),
    messages=messages,
    tools=tools,
)

# Step 2: Model decides to call function(s)
print(completion.model_dump())

# Step 3: Execute search_db function
def call_function(name, args):
    if name == "search_db":
        return search_db(**args)


for tool_call in completion.choices[0].message.tool_calls:
    name = tool_call.function.name
    args = json.loads(tool_call.function.arguments)
    messages.append(completion.choices[0].message)

    result = call_function(name, args)
    messages.append(
        {"role": "tool", "tool_call_id": tool_call.id, "content": json.dumps(result)}
    )

# Step 4: Supply result and call model again
class DBResponse(BaseModel):
    answer: str = Field(description="The answer to the user's question.")
    source: int = Field(description="The record id of the answer.")


completion_2 = client.beta.chat.completions.parse(
    model=os.getenv("MODEL_NAME", "gpt-4.1"),
    messages=messages,
    tools=tools,
    response_format=DBResponse,
)

# Step 5: Check model response
final_response = completion_2.choices[0].message.parsed
print(final_response.answer)
print(final_response.source)

# Question that doesn't trigger the tool
messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": "What is the weather in Tokyo?"},
]

completion_3 = client.beta.chat.completions.parse(
    model=os.getenv("MODEL_NAME", "gpt-4.1"),
    messages=messages,
    tools=tools,
)

print(completion_3.choices[0].message.content)
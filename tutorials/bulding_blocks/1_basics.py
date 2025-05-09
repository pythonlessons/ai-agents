import os
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


completion = client.chat.completions.create(
    model=os.getenv("MODEL_NAME", "gpt-4.1"),
    messages=[
        {"role": "system", "content": "You're a helpful assistant."},
        {
            "role": "user",
            "content": "Write a five-line poem about the Python programming language.",
        },
    ],
)

response = completion.choices[0].message.content
print(response)
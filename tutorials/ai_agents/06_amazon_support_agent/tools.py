import os
import openai
from typing import Dict, Any


class AmazonPolicyTool:
    def __init__(self, name: str = "get_amazon_policy", model: str="gpt-4.1"):
        self.name = name
        self.model = model  # Specify the model to use

        self.client = openai.Client(
            api_key=os.getenv("VECTORIZE_IO_API_KEY"),  # or use your token directly
            base_url="https://api.vectorize.io/v1/org/c9330a9b-3a05-4ea3-af81-802bce4c63da/pipelines/aip48281-7f36-42b1-beb4-96ed44e9ef65"
        )

    def __call__(self, query: str) -> Dict[str, Any]:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": query}],
        )
        return {"result": response.choices[0].message.content}

    def tool(self):
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": "Get answers from Amazon's return/shipping and selling policy documents.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Question about Amazon policies"},
                    },
                    "required": ["query"],
                },
            },
        }


class EmailSupportTool:
    def __init__(self, name: str = "email_amazon_support"):
        self.name = name

    def __call__(self, message: str) -> Dict[str, Any]:
        print(f"\n[Simulated Email] 📧 Sending the following message to support:\n{message}\n")
        return {"result": f"Email sent to Amazon support with message: {message}"}

    def tool(self):
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": "Send a message to Amazon support (simulated).",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "message": {"type": "string", "description": "Message to send to support"},
                    },
                    "required": ["message"],
                },
            },
        }


class ConfirmTool:
    def __init__(self, name: str = "confirm_action"):
        self.name = name

    def __call__(self, prompt: str) -> Dict[str, Any]:
        while True:
            user_input = input(f"\n🤖 {prompt} (yes/no): ").strip().lower()
            if user_input in {"yes", "no"}:
                return {"confirmation": user_input}
            print("Please type 'yes' or 'no'.")

    def tool(self):
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": "Ask the user to confirm an action with yes or no.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "prompt": {"type": "string", "description": "Prompt to show the user for confirmation"},
                    },
                    "required": ["prompt"],
                },
            },
        }
    

class ImproveMessageTool:
    def __init__(self, name: str = "improve_message", model: str = "gpt-4.1"):
        self.name = name
        self.model = model
        self.client = openai.Client(
            api_key=os.getenv("OPENAI_API_KEY"),  # Using standard OpenAI endpoint
            base_url=os.getenv("OPENAI_API_BASE_URL", "https://api.openai.com/v1")
        )

    def __call__(self, message: str) -> Dict[str, Any]:
        prompt = f"Improve the following message for clarity and professionalism:\n\n{message}"
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
        )
        improved = response.choices[0].message.content
        print(f"\n🤖 Suggested improved message:\n{improved}\n")
        while True:
            confirm = input("Do you want to use the improved message? (yes/no): ").strip().lower()
            if confirm in {"yes", "no"}:
                break
        return {"improved_message": improved if confirm == "yes" else message}

    def tool(self):
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": "Improve an email message for professionalism before sending.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "message": {"type": "string", "description": "The email message to improve"},
                    },
                    "required": ["message"],
                },
            },
        }
import os
import json
from typing import Dict, Any, List, Union, Optional
from openai import OpenAI
import vectorize_client as v, os
from pydantic import BaseModel, Field
from datetime import datetime
import numpy as np

class RetrievalTool:
    def __init__(self, token: str, organization_id: str, pipeline_id: str, return_all_data: bool=False, name: str = "retrieve_documents", host="https://api.vectorize.io/v1"):
        self.name = name
        self.organization_id = organization_id
        self.pipeline_id = pipeline_id
        self.return_all_data = return_all_data
        self.api = v.ApiClient(v.Configuration(access_token=token, host=host))

    def __call__(self, question: str) -> Dict[str, Any]:
        # Use the provided question as-is and retrieve relevant documents
        pipelines = v.PipelinesApi(self.api)
        response = pipelines.retrieve_documents(
            self.organization_id,
            self.pipeline_id,
            v.RetrieveDocumentsRequest(
                question="*" if self.return_all_data else question,
                num_results=100,     # up to the API limit
                rerank=True
            )
        )

        context = [json.dumps(doc.additional_properties) for doc in response.documents]
        # Return a structured JSON object so the LLM can pass these to the analyzer
        return context
    
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
                "description": "Retrieve relevant documents for the user's question. Always call this tool FIRST to gather context before any analysis.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "question": {"type": "string", "description": "The user's question to retrieve supporting context for."},
                    },
                    "required": ["question"],
                    "additionalProperties": False,
                },
                "strict": True,
            },
        }
    
class AnalyzerTool:
    def __init__(self, token: str, base_url: str, model: str='gpt-5', name: str = "analyze_documents"):
        self.name = name
        self.client = OpenAI(api_key=token, base_url=base_url)
        self.model = model

    def __call__(self, query: str, context: Union[List[str], str]) -> Dict[str, Any]:
        # Guard: require non-empty context
        if not context or (isinstance(context, list) and len(context) == 0):
            return {"error": "No context provided. Please call retrieve_documents first and pass its context to analyze_documents.", "result": None}

        # Normalize context to a readable string
        if isinstance(context, list):
            context_str = "\n\n".join(context)
        else:
            context_str = str(context)

        system_prompt = "You are a helpful assistant that can answer questions about purchases from receipts."
        prompt = f"""
        You are tasked with answering a question using provided chunks of information. Your goal is to provide an accurate answer.
        
        Question:
        {query}

        Context (excerpts from receipts and related data):
        {context_str}

        Answer the question using only the information provided in the context. If the answer is not in the context, say: "Sorry I don't know".
        """

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
        )
        return {"result": response.choices[0].message.content}

    def tool(self):
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": "Analyze documents to answer the user's question. Call this ONLY AFTER retrieve_documents and pass its context along with the original query.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "The original user question."},
                        "context": {"type": "array", "items": {"type": "string"}, "description": "Text chunks returned from retrieve_documents."},
                    },
                    "required": ["query", "context"],
                    "additionalProperties": False,
                },
                "strict": True,
            },
        }

class ColumnSchema(BaseModel):
    name: str = Field(description="The name of the column")
    dtype: str = Field(description="The data type of the column (string|number|date)")
    description: str = Field(description="A brief description of the column")

class TableSchema(BaseModel):
    columns: List[ColumnSchema] = Field(
        description="List of columns in the table schema"
    )

class InferTableSchemaTool:
    def __init__(self, token: str, base_url: str, model: str='gpt-5', name: str = "infer_table_schema"):
        self.name = name
        self.client = OpenAI(api_key=token, base_url=base_url)
        self.model = model

    def __call__(self, analysis_text: str) -> Dict[str, Any]:
        system_prompt = "You convert summarized spending text into a tabular schema."
        user_prompt = f"""
        From the following analysis text, infer an appropriate table schema.
        Return ONLY the structured object adhering to the provided schema.
        
        Analysis text:
        {analysis_text}
        """
        resp = self.client.beta.chat.completions.parse(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format=TableSchema,
        )
        parsed = resp.choices[0].message.parsed
        # Return as a plain dict for the agent/tool message
        return parsed.model_dump()

    def tool(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": "Infer a tabular schema (columns) from a natural-language analysis of spending.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "analysis_text": {"type": "string", "description": "The analysis text to convert into a table schema."}
                    },
                    "required": ["analysis_text"],
                    "additionalProperties": False,
                },
                "strict": True,
            },
        }

class BuildDataFrameTool:
    def __init__(self, token: str, base_url: str, model: str='gpt-5', name: str = "build_dataframe"):
        self.name = name
        self.client = OpenAI(api_key=token, base_url=base_url)
        self.model = model

    class RowsModel(BaseModel):
        rows: List[List[Union[str, float, int, None]]] = Field(
            description="2D array where each inner list is a row matching the provided columns order"
        )

    def __call__(self, analysis_text: str, columns: List[Dict[str, str]]) -> Dict[str, Any]:
        # Prepare prompt
        col_names = [c.get("name") for c in columns]
        system_prompt = "You extract structured rows from spending summaries. Return only data matching the schema."
        user_prompt = f"""
        Based on the analysis text, extract rows matching these columns in order: {col_names}.
        Return ONLY the structured object adhering to the provided schema.
        
        Analysis text:
        {analysis_text}
        """
        resp = self.client.chat.completions.parse(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format=self.RowsModel,
        )
        parsed = resp.choices[0].message.parsed
        rows = parsed.rows if hasattr(parsed, "rows") else []

        # Build DataFrame if pandas available
        try:
            import pandas as pd
            df = pd.DataFrame(rows, columns=col_names)
            # Optional: cast dtypes according to provided columns metadata
            for i, col in enumerate(columns):
                dtype = (col.get("dtype") or "").lower()
                if dtype in ("number", "float", "int"):
                    numeric = pd.to_numeric(df.iloc[:, i], errors='coerce')
                    # Replace infinities with NaN (avoid deprecated use_inf_as_na option)
                    numeric.replace([np.inf, -np.inf], np.nan, inplace=True)
                    df.iloc[:, i] = numeric
                elif dtype == "date":
                    df.iloc[:, i] = pd.to_datetime(df.iloc[:, i], errors='coerce')
                # else leave as object/string
            df_text = df.to_string(index=False)
            print(df)
            return {"columns": col_names, "rows": rows, "dataframe": df_text}
        except Exception as e:
            return {"columns": col_names, "rows": rows, "error": f"pandas not available or failed to build DataFrame: {e}"}

    def tool(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": "Create a pandas DataFrame from analysis text using a provided schema. Returns columns and rows for saving.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "analysis_text": {"type": "string", "description": "The analysis text to convert to rows."},
                        "columns": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "name": {"type": "string"},
                                    "dtype": {"type": "string"},
                                    "description": {"type": "string"},
                                },
                                "required": ["name", "dtype", "description"],
                                "additionalProperties": False,
                            },
                            "description": "Column definitions as returned by infer_table_schema."
                        },
                    },
                    "required": ["analysis_text", "columns"],
                    "additionalProperties": False,
                },
                "strict": True,
            },
        }

class SaveDataFrameTool:
    def __init__(self, name: str = "save_dataframe"):
        self.name = name

    def _make_unique_path(self, path: str, fmt: str) -> str:
        """Ensure the target file path is unique by appending a timestamp (and counter if needed)."""
        root, ext = os.path.splitext(path)
        if not ext:
            ext = f".{fmt}"
        candidate = f"{root}{ext}"
        if not os.path.exists(candidate):
            return candidate
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        candidate = f"{root}_{ts}{ext}"
        if not os.path.exists(candidate):
            return candidate
        i = 1
        while True:
            candidate_i = f"{root}_{ts}_{i}{ext}"
            if not os.path.exists(candidate_i):
                return candidate_i
            i += 1

    def __call__(self, columns: List[Dict[str, str]], rows: List[List[Union[str, float, int, None]]], path: Optional[str] = None, fmt: str = "csv") -> Dict[str, Any]:
        try:
            import pandas as pd
        except Exception as e:
            return {"saved": False, "error": f"pandas not available: {e}"}

        if not path:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = os.path.join(os.getcwd(), f"receipts_table_{ts}.{fmt}")
        else:
            path = os.path.abspath(path)

        # Ensure directory exists
        os.makedirs(os.path.dirname(path), exist_ok=True)

        # Ensure unique filename if a file with the same name already exists
        path = self._make_unique_path(path, fmt)

        # Build DataFrame
        col_names = [c.get("name") for c in columns]
        df = pd.DataFrame(rows, columns=col_names)
        for i, col in enumerate(columns):
            dtype = (col.get("dtype") or "").lower()
            if dtype in ("number", "float", "int"):
                df.iloc[:, i] = pd.to_numeric(df.iloc[:, i], errors='coerce')
            elif dtype == "date":
                df.iloc[:, i] = pd.to_datetime(df.iloc[:, i], errors='coerce')

        try:
            if fmt.lower() == "csv":
                df.to_csv(path, index=False)
            elif fmt.lower() == "json":
                df.to_json(path, orient='records')
            elif fmt.lower() == "parquet":
                df.to_parquet(path, index=False)
            else:
                return {"saved": False, "error": f"Unsupported format: {fmt}"}
            return {"saved": True, "path": path, "format": fmt.lower()}
        except Exception as e:
            return {"saved": False, "error": str(e), "path": path}

    def tool(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": "Save a DataFrame to disk from provided columns and rows. Supports csv, json, parquet. If no path is provided, a timestamped filename is generated. If the target path already exists, a unique timestamp suffix is appended to avoid overwriting.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "columns": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "name": {"type": "string"},
                                    "dtype": {"type": "string"},
                                    "description": {"type": "string"},
                                },
                                "required": ["name", "dtype", "description"],
                                "additionalProperties": False,
                            },
                        },
                        "rows": {
                            "type": "array",
                            "items": {
                                "type": "array",
                                "items": {"type": ["string", "number", "null"]}
                            }
                        },
                        "path": {"type": "string", "description": "Absolute or relative file path to save to (optional). Pass empty string to auto-generate. If this path already exists, a unique timestamp suffix will be added."},
                        "fmt": {"type": "string", "enum": ["csv", "json", "parquet"], "description": "Output format. Defaults to csv."}
                    },
                    "required": ["columns", "rows", "path", "fmt"],
                    "additionalProperties": False,
                },
                "strict": True,
            },
        }
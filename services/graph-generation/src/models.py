from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
from llama_index.core import Settings
from .config import config

class Models:
    @staticmethod
    def initialize_models():
        embed_model = AzureOpenAIEmbedding(
            model="text-embedding-ada-002",
            deployment_name="text-embedding-ada-002",
            api_key=config.azure_openai_api_key,
            azure_endpoint=config.azure_openai_endpoint,
            api_version="2024-02-01",
        )
        llm = AzureOpenAI(
            engine="gpt-4o", 
            model="gpt-4o", 
            temperature=0.0,
            api_key=config.azure_openai_api_key,
            azure_endpoint=config.azure_openai_endpoint,
            api_version="2023-09-01-preview", 
        )
        Settings.llm = llm
        Settings.embed_model = embed_model
        return llm, embed_model

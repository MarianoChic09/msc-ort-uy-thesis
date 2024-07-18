from dotenv import find_dotenv, load_dotenv
from pydantic_settings import BaseSettings
from typing import Optional
from loguru import logger
# load my .env file variables as environment variables so pydantic_settings can access them
# to create the Config object
load_dotenv(find_dotenv()) #

import os
os.environ.get("DEVELOPMENT", None)

if os.environ.get("DEVELOPMENT", None) == "True":
    logger.info("Secrets loaded from .env")

    # load_dotenv(find_dotenv()) #
    pass
else:
    logger.info("Loading secrets from Azure Key Vault")

    from azure.identity import DefaultAzureCredential
    from azure.keyvault.secrets import SecretClient

    key_vault_uri = os.environ["keyvaulturi"]
    credential = DefaultAzureCredential()
    logger.info(f"Credential: {credential}")
    client = SecretClient(vault_url=key_vault_uri, credential=credential)



# load_dotenv(find_dotenv())


class Config(BaseSettings):
    """
    Configuration settings for the agent service

    Attributes:
        azure_openai_key: The API key for the Azure OpenAI service
        azure_openai_endpoint: The endpoint for the Azure OpenAI service
    
    Values are read from environment variables.
    If they are not found there, default values are used.
    If the DEVELOPMENT environment variable is set to True, the values are read from the .env file.
    If the DEVELOPMENT environment variable is set to False, the values are read from Azure Key Vault.
    """
    azure_openai_api_key: str = None
    azure_openai_endpoint: str = None
    embeddings_model_name: str = "text-embedding-ada-002"
    azure_openai_embeddings_api_version: str = "2024-02-01"


    if os.environ.get("DEVELOPMENT", None) == "True":
        pass
    else:
        azure_openai_api_key: str = client.get_secret("azure-openai-key").value
        azure_openai_endpoint: str = client.get_secret("azure-openai-endpoint").value
        

config = Config()

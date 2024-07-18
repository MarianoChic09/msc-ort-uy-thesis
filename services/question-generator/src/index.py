from llama_index.core import StorageContext, load_index_from_storage
from .models import Models
import nest_asyncio

nest_asyncio.apply()

class IndexManager:

    @staticmethod
    def load_index(storage_dir="./storage"):
        storage_context = StorageContext.from_defaults(persist_dir=storage_dir)
        return load_index_from_storage(storage_context)
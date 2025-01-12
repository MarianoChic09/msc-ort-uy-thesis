from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext, load_index_from_storage

class IndexManager:
    @staticmethod
    def create_index(documents, storage_dir, storage_context=None):
        index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)
        index.storage_context.persist(storage_dir)
        return index

    @staticmethod
    def load_documents(directory_path):
        return SimpleDirectoryReader(directory_path).load_data()

    @staticmethod
    def load_index(storage_dir):
        storage_context = StorageContext.from_defaults(persist_dir=storage_dir)
        return load_index_from_storage(storage_context)

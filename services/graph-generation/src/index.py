from llama_index.core import PropertyGraphIndex, SimpleDirectoryReader, StorageContext, load_index_from_storage
from llama_index.core.indices.property_graph import SchemaLLMPathExtractor
from .models import Models
from .schemas import entities, relations, validation_schema
import nest_asyncio

nest_asyncio.apply()

class IndexManager:
    @staticmethod
    def create_schema_llm_extractor(llm):
        return SchemaLLMPathExtractor(
            llm=llm,
            possible_entities=entities,
            possible_relations=relations,
            kg_validation_schema=validation_schema,
            strict=True,
        )

    # @staticmethod
    # def create_index(documents, llm, embed_model, kg_extractor):
    #     index = PropertyGraphIndex.from_documents(
    #         documents,
    #         llm=llm,
    #         embed_model=embed_model,
    #         kg_extractors=[kg_extractor],
    #         show_progress=True,
    #     )
    #     index.storage_context.persist("./storage")
    #     return index
    
    @staticmethod
    def create_index(documents, llm, embed_model, kg_extractor=None, storage_dir="./storage", graph_name=None):
        index = PropertyGraphIndex.from_documents(
            documents,
            llm=llm,
            embed_model=embed_model,
            kg_extractors=[kg_extractor] if kg_extractor else None,
            show_progress=True,
        )
        if graph_name:
            index.property_graph_store.save_networkx_graph(name=graph_name)
        index.storage_context.persist(storage_dir)
        return index
    @staticmethod
    def load_documents(directory_path):
        return SimpleDirectoryReader(directory_path).load_data()
    
    @staticmethod
    def load_index(storage_dir="./storage"):
        storage_context = StorageContext.from_defaults(persist_dir=storage_dir)
        return load_index_from_storage(storage_context)
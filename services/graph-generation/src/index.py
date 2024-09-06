from llama_index.core import (
    PropertyGraphIndex,
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage,
    VectorStoreIndex,
)
from llama_index.core.indices.property_graph import SchemaLLMPathExtractor
from llama_index.core.indices.property_graph import DynamicLLMPathExtractor
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine

from .models import Models
from loguru import logger
from typing import Literal

# from .schemas import entities, relations, validation_schema
import nest_asyncio

nest_asyncio.apply()
from typing import Literal, Optional
from pydantic import BaseModel


class Query(BaseModel):
    query: str
    num_docs: Optional[int]


class IndexManager:
    @staticmethod
    def create_schema_llm_extractor(llm, entities, relations, validation_schema):
        logger.debug(f"entities type: {type(entities)}")  # Should print <class 'list'>
        logger.debug(
            f"relations type: {type(relations)}"
        )  # Should print <class 'list'>
        logger.debug(
            f"validation_schema type: {type(validation_schema)}"
        )  # Should print <class 'dict'>

        # return DynamicLLMPathExtractor(
        #     llm=llm,
        #     max_triplets_per_chunk=20,
        #     num_workers=4,
        #     allowed_entity_types=entities,
        #     allowed_relation_types=relations,
        #     validation_schema=validation_schema,
        # )
        EntityType = Literal[tuple(entities)]
        RelationType = Literal[tuple(relations)]
        return SchemaLLMPathExtractor(
            llm=llm,
            possible_entities=EntityType,
            possible_relations=RelationType,
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
    def create_index(
        documents,
        llm,
        embed_model,
        kg_extractor=None,
        storage_dir="./storage",
        graph_name=None,
    ):
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

    @staticmethod
    def create_baseline_rag_index(
        documents,
        llm,
        embed_model,
        storage_dir="./storage",
    ):
        index = VectorStoreIndex.from_documents(
            documents,
            llm=llm,
            embed_model=embed_model,
            show_progress=True,
        )
        index.storage_context.persist(storage_dir)
        return index

    @staticmethod
    def query_baseline_rag_index(query, num_retrieved_docs=5):
        index = IndexManager.load_index(storage_dir="./baseline_rag_storage")

        # configure retriever
        retriever = VectorIndexRetriever(
            index=index,
            similarity_top_k=num_retrieved_docs,
        )

        query_engine = RetrieverQueryEngine(
            retriever=retriever,
        )

        # query_engine = index.as_query_engine()
        response = query_engine.query(query)
        return response

    @staticmethod
    def query_graph_rag_free_form(query):
        index = IndexManager.load_index(storage_dir="./baseline_rag_storage")

        # configure retriever
        retriever = VectorIndexRetriever(
            index=index,
            similarity_top_k=10,
        )

        query_engine = RetrieverQueryEngine(
            retriever=retriever,
        )

        # query_engine = index.as_query_engine()
        response = query_engine.query(query)
        return response

    @staticmethod
    def query_index(query: str, storage_dir, num_docs: int = 5):
        try:
            index = IndexManager.load_index(storage_dir)
        except Exception as e:
            # raise HTTPException(status_code=500, detail=f"Failed to load index: {e}")
            raise Exception(f"Failed to load index: {e}")

        query_engine = index.as_query_engine(
            include_text=True,
            similarity_top_k=num_docs,
            # embed_model=embed_model,
        )
        triplets = index.property_graph_store.get_triplets()
        logger.info(f"Triplets: {triplets}")
        response = query_engine.query(query)
        logger.info(f"Response: {response}")
        return response

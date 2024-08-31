from llama_index.core import (
    PropertyGraphIndex,
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage,
)
from llama_index.core.indices.property_graph import SchemaLLMPathExtractor
from llama_index.core.indices.property_graph import DynamicLLMPathExtractor

from .models import Models
from loguru import logger
from typing import Literal

# from .schemas import entities, relations, validation_schema
import nest_asyncio

nest_asyncio.apply()


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

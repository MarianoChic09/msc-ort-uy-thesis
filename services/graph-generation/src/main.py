from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from .models import Models
from .index import IndexManager
from .utils import save_uploaded_file
from .schemas import Query, SchemaExtractorRequest, get_default_schema
from loguru import logger
import gc

app = FastAPI()


@app.on_event("startup")
async def startup_event():
    global llm, embed_model
    llm, embed_model = Models.initialize_models()


def process_document(
    file: UploadFile,
    storage_dir,
    graph_name=None,
    kg_extractor=None,
    baseline_rag=False,
):
    file_location = f"data/03_processed/{file.filename}"
    save_uploaded_file(file, file_location)

    documents = IndexManager.load_documents("data/03_processed/")

    if baseline_rag:
        index = IndexManager.create_baseline_rag_index(
            documents, llm, embed_model, storage_dir
        )
    else:
        index = IndexManager.create_index(
            documents, llm, embed_model, kg_extractor, storage_dir, graph_name
        )
    return index


@app.post("/upload-document")
async def upload_document(file: UploadFile = File(...)):
    kg_extractor = IndexManager.create_schema_llm_extractor(llm)
    process_document(
        file,
        storage_dir="./storage",
        graph_name="data/08_reports/kg_predefined_schema.html",
        kg_extractor=kg_extractor,
    )
    return {"message": "Document uploaded and index created successfully."}


@app.post("/predefined-schema-extractor")
async def predefined_schema_extractor(
    file: UploadFile = File(...),
    request: SchemaExtractorRequest = Depends(get_default_schema),
):
    logger.info(f"Request: {request}")
    logger.info(f"Entities: {request.entities}")
    logger.info(f"Relations: {request.relations}")
    logger.info(f"Validation Schema: {request.validation_schema}")
    logger.debug(f"The type of request is: {type(request)}")
    logger.debug(f"The type of request.entities is: {type(request.entities)}")
    logger.debug(f"The type of request.relations is: {type(request.relations)}")
    logger.debug(
        f"The type of request.validation_schema is: {type(request.validation_schema)}"
    )

    kg_extractor = IndexManager.create_schema_llm_extractor(
        llm,
        request.entities,
        request.relations,
        request.validation_schema,
    )
    process_document(
        file,
        storage_dir="./storage_defensa_test",
        graph_name="data/08_reports/kg_predefined_schema_defensa_test.html",
        kg_extractor=kg_extractor,
    )
    # Dereference the extractor after use
    kg_extractor = None

    # Optionally force garbage collection
    gc.collect()

    return {"message": "Document uploaded and index created successfully."}


@app.post("/free-form-extractor")
async def free_form_extractor(file: UploadFile = File(...)):
    process_document(
        file,
        storage_dir="./free_form_storage",
        graph_name="data/08_reports/kg_free_form.html",
    )
    return {"message": "Document uploaded and free form index created successfully."}


@app.post("/baseline-rag")
async def baseline_rag(file: UploadFile = File(...)):
    process_document(file, storage_dir="./baseline_rag_storage", baseline_rag=True)
    return {"message": "Document uploaded and baseline RAG index created successfully."}


from pydantic import BaseModel


class QueryRequest(BaseModel):
    query: str
    num_retrieved_docs: int = 5


@app.post("/query-baseline-rag")
async def query_baseline_rag(request: QueryRequest):
    response = IndexManager.query_baseline_rag_index(
        request.query, request.num_retrieved_docs
    )
    return response


class QueryRequest(BaseModel):
    query: str
    num_docs: int = 5
    # num_retrieved_docs: int = 5


# @app.post("/query-graph-rag-free-form")
# async def query_graph_rag_free_form(request: QueryRequest):
#     response = IndexManager.query_graph_rag_free_form(
#         request.query, request.num_retrieved_docs
#     )
#     return response


@app.post("/query-free-form")
async def query_free_form_index_endpoint(query_request: QueryRequest):
    return IndexManager.query_index(
        query=query_request.query,
        storage_dir="./free_form_storage",
        num_docs=query_request.num_docs,
    )


@app.post("/query-schema-guided")
async def query_index_endpoint(query_request: QueryRequest):
    logger.info(f"query_requests: {query_request}")
    return IndexManager.query_index(
        query=query_request.query,
        storage_dir="./storage",
        num_docs=query_request.num_docs,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)

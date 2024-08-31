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


def process_document(file: UploadFile, storage_dir, graph_name=None, kg_extractor=None):
    file_location = f"data/03_processed/{file.filename}"
    save_uploaded_file(file, file_location)

    documents = IndexManager.load_documents("data/03_processed/")
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
        storage_dir="./storage",
        graph_name="data/08_reports/kg_predefined_schema.html",
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


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)

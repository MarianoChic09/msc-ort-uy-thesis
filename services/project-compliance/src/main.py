from fastapi import FastAPI, UploadFile, File, HTTPException
from .question_generator import get_questions
from .index import IndexManager
from .utils import ensure_directory_exists
from .schemas import QueryRequest
from .models import Models
from llama_index.core import StorageContext
import os
import logging

app = FastAPI()
# Configuración del logger
logger = logging.getLogger("uvicorn")
logger.setLevel(logging.INFO)


# @app.on_event("startup")
# async def startup_event():
#     global index, storage_context
#     documents = IndexManager.load_documents("/data/03_processed")
#     storage_context = StorageContext.from_defaults(persist_dir="./storage")
#     index = IndexManager.create_index(documents, storage_context=storage_context)
@app.on_event("startup")
async def startup_event():
    global index, storage_context
    global llm, embed_model
    llm, embed_model = Models.initialize_models()
    storage_dir = "./storage_vector_store"
    try:
        if not os.path.exists(os.path.join(storage_dir, "docstore.json")):
            raise FileNotFoundError
        index = IndexManager.load_index(storage_dir)
    except FileNotFoundError:
        index = None
        logger.info("Index not found. Creating a new index...")
        await create_index()


@app.post("/upload-document")
async def upload_document(file: UploadFile = File(...)):
    file_location = f"./data/03_processed/{file.filename}"
    ensure_directory_exists(os.path.dirname(file_location))
    with open(file_location, "wb+") as f:
        f.write(file.file.read())
    # Actualizar el índice con el nuevo documento
    documents = IndexManager.load_documents("./data/03_processed")
    storage_dir = "./storage_vector_store"
    storage_context = StorageContext.from_defaults(persist_dir=storage_dir)

    index = IndexManager.create_index(
        documents, storage_context=storage_context, storage_dir=storage_dir
    )

    return {
        "message": f"Document uploaded and index updated successfully. Index ID: {index.index_id}"
    }


# @app.post("/create-index")
async def create_index(storage_dir="./storage_vector_store"):
    global index, storage_context
    documents = IndexManager.load_documents("./data/03_processed")
    # ensure_directory_exists(storage_dir)
    # storage_context = StorageContext.from_defaults(persist_dir=storage_dir)

    index = IndexManager.create_index(documents=documents, storage_dir=storage_dir)

    return {"message": "Index created and documents indexed successfully"}


@app.post("/generate-questions")
async def generate_questions(query_request: QueryRequest):
    prompt = query_request.query
    questions = get_questions(prompt)
    return {"questions": questions}


@app.post("/query-project")
async def query(query_request: QueryRequest):
    # index = IndexManager.load_index("./storage_vector_store")
    if not index:
        raise HTTPException(
            status_code=500, detail="Index not loaded. Please create the index first."
        )

    query_engine = index.as_query_engine(include_text=True, similarity_top_k=6)
    response = query_engine.query(query_request.query)
    return {"response": response}


@app.post("/search-answers-to-questions")
async def search_answers(query_request: QueryRequest):
    # index = IndexManager.load_index("./storage_vector_store")
    if not index:
        raise HTTPException(
            status_code=500, detail="Index not loaded. Please create the index first."
        )

    questions = get_questions(query_request.query)
    results = []
    for question in questions:
        query_engine = index.as_query_engine(include_text=True, similarity_top_k=2)
        response = query_engine.query(question)
        results.append({"question": question, "answer": response})
    return {"results": results}


@app.get("/")
async def read_root():
    return {"message": "Welcome to the Project Compliance API!"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8540)

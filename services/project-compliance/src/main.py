from fastapi import FastAPI, UploadFile, File, HTTPException
from .question_generator import get_questions
from .index import IndexManager
from .utils import ensure_directory_exists
from .schemas import QueryRequest
from llama_index.core import StorageContext
import os 

app = FastAPI()

# @app.on_event("startup")
# async def startup_event():
#     global index, storage_context
#     documents = IndexManager.load_documents("/data/03_processed")
#     storage_context = StorageContext.from_defaults(persist_dir="./storage")
#     index = IndexManager.create_index(documents, storage_context=storage_context)
@app.on_event("startup")
async def startup_event():
    global index, storage_context
    storage_dir = "./storage"
    try:
        index = IndexManager.load_index(storage_dir)
    except Exception as e:
        # Si no se puede cargar el índice, crear uno nuevo
        documents = IndexManager.load_documents("./data/03_processed")
        ensure_directory_exists(storage_dir)
        storage_context = StorageContext.from_defaults(persist_dir=storage_dir)
        index = IndexManager.create_index(documents, storage_context=storage_context)

@app.post("/upload-document/")
async def upload_document(file: UploadFile = File(...)):
    file_location = f"/data/03_processed/{file.filename}"
    ensure_directory_exists(os.path.dirname(file_location))
    with open(file_location, "wb+") as f:
        f.write(file.file.read())
    return {"message": "Document uploaded successfully"}

@app.post("/generate-questions/")
async def generate_questions(query_request: QueryRequest):
    prompt = query_request.query
    questions = get_questions(prompt)
    return {"questions": questions}

@app.post("/search-answers/")
async def search_answers(query_request: QueryRequest):
    questions = get_questions(query_request.query)
    results = []
    for question in questions:
        query_engine = index.as_query_engine(
            include_text=True,
            similarity_top_k=5,
        )
        response = query_engine.query(question)
        results.append({
            "question": question,
            "answer": response
        })
    return {"results": results}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=9000)

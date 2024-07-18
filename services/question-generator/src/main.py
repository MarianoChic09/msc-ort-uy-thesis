from fastapi import FastAPI, UploadFile, File, HTTPException
from .models import Models
from .index import IndexManager
from .schemas import Query

app = FastAPI()

@app.on_event("startup")
async def startup_event():
    global llm, embed_model
    llm, embed_model = Models.initialize_models()

def query_index(query_request: Query, storage_dir):
    try:
        index = IndexManager.load_index(storage_dir)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load index: {e}")

    query_engine = index.as_query_engine(
        include_text=True,
        similarity_top_k=5,
        # embed_model=embed_model,
    )
    response = query_engine.query(query_request.query)
    return {"response": response}

@app.post("/query")
async def query_index_endpoint(query_request: Query):
    return query_index(query_request, storage_dir="./storage")

@app.post("/query-free-form")
async def query_free_form_index_endpoint(query_request: Query):
    return query_index(query_request, storage_dir="./free_form_storage")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=6000)
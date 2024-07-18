from fastapi import FastAPI, UploadFile, File, HTTPException
from .models import Models
from .index import IndexManager
from .utils import save_uploaded_file
from .schemas import Query

app = FastAPI()

@app.on_event("startup")
async def startup_event():
    global llm, embed_model
    llm, embed_model = Models.initialize_models()

@app.post("/upload-document/")
async def upload_document(file: UploadFile = File(...)):
    file_location = f"data/03_processed/{file.filename}"
    save_uploaded_file(file, file_location)

    documents = IndexManager.load_documents("data/03_processed/")
    kg_extractor = IndexManager.create_schema_llm_extractor(llm)
    index = IndexManager.create_index(documents, llm, embed_model, kg_extractor)
    index.property_graph_store.save_networkx_graph(name="./data/08_reports/kg_result.html")

@app.post("/query/")
async def query_index(query_request: Query):
    try:
        index = IndexManager.load_index()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load index: {e}")

    query_engine = index.as_query_engine(
        include_text=True,
        similarity_top_k=2,
        embed_model=embed_model,
    )
    response = query_engine.query(query_request.query)

    return {"response": response}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

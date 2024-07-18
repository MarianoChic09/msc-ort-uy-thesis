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

def process_document(file: UploadFile, storage_dir, graph_name=None, kg_extractor=None):
    file_location = f"data/03_processed/{file.filename}"
    save_uploaded_file(file, file_location)
    
    documents = IndexManager.load_documents("data/03_processed/")
    index = IndexManager.create_index(documents, llm, embed_model, kg_extractor, storage_dir, graph_name)
    return index

@app.post("/upload-document")
async def upload_document(file: UploadFile = File(...)):
    kg_extractor = IndexManager.create_schema_llm_extractor(llm)
    process_document(file, storage_dir="./storage", graph_name="data/08_reports/kg_predefined_schema.html", kg_extractor=kg_extractor)
    return {"message": "Document uploaded and index created successfully."}

@app.post("/free-form-extractor")
async def free_form_extractor(file: UploadFile = File(...)):
    process_document(file, storage_dir="./free_form_storage", graph_name="data/08_reports/kg_free_form.html")
    return {"message": "Document uploaded and free form index created successfully."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

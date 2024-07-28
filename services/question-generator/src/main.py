from fastapi import FastAPI, UploadFile, File, HTTPException
from .models import Models
from .index import IndexManager
from .schemas import Query
import logging 

app = FastAPI()
logger = logging.getLogger("uvicorn")
logger.setLevel(logging.INFO)

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

@app.get("/questions-from-standard")
async def get_questions_from_standard():
    # This endpoint uses a prompt to generate questions from the GraphRAG 
    # db. The document is a standard. The prompt is meant to generate a list 
    # of questions, parse them and return them as a list of strings in the 
    # JSON response.

    
    prompt = """As an expert in environmental compliance, you are tasked
            with ensuring that a project adheres to all relevant environmental 
            regulations and guidelines. Below is a summary of the borrower's 
            responsibilities related to environmental compliance. Based on this
            information, generate a comprehensive list of questions that you 
            would ask to validate the project's compliance. Your questions 
            should cover all aspects mentioned in the summary. Ensure that 
            the questions are thorough and specific enough to identify any
            potential compliance issues."""
    
    response = query_index(Query(query=prompt), storage_dir="./free_form_storage")
    
    # Now we need to parse the response and extract the questions
    # from the generated text
    logger.info(response)

    response_text = response['response'].response
    questions = response_text.split("\n")

    # Discard empty strings and strings with less than 5 characters
    questions = [q for q in questions if len(q) > 5]

    # Discard strings that doesn't contain a question mark
    questions = [q for q in questions if "?" in q]

    return {"questions": questions}
    



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=6000)
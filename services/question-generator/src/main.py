from fastapi import FastAPI, UploadFile, File, HTTPException
from .models import Models
from .index import IndexManager
from .schemas import Query
import logging
import requests

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
            potential compliance issues.
            Treat every question generated as a separate entity."""

    response = query_index(Query(query=prompt), storage_dir="./free_form_storage")

    # Now we need to parse the response and extract the questions
    # from the generated text
    logger.info(response)

    response_text = response["response"].response
    questions = response_text.split("\n")

    # Discard empty strings and strings with less than 5 characters
    questions = [q for q in questions if len(q) > 5]

    # Discard strings that doesn't contain a question mark
    questions = [q for q in questions if "?" in q]

    return {"questions": questions}


def get_environmental_compliance_prompt():
    return """As an expert in environmental compliance, you are tasked
            with ensuring that a project adheres to all relevant environmental 
            regulations and guidelines. Below is a summary of the borrower's 
            responsibilities related to environmental compliance. Based on this
            information, generate a comprehensive list of questions that you 
            would ask to validate the project's compliance. Your questions 
            should cover all aspects mentioned in the summary. Ensure that 
            the questions are thorough and specific enough to identify any
            potential compliance issues.
            Treat every question generated as a separate entity."""


def query_endpoint(url, prompt, num_retrieved_docs=5):
    body = {"query": prompt, "num_retrieved_docs": num_retrieved_docs}
    response = requests.post(url, json=body)
    return response.json()


def extract_questions(response_text):
    questions = response_text.split("\n")
    questions = [q for q in questions if len(q) > 5 and "?" in q]
    return questions


async def generate_questions(endpoint_url):
    prompt = get_environmental_compliance_prompt()
    response_json = query_endpoint(endpoint_url, prompt)
    logger.info(response_json)

    questions = extract_questions(response_json["response"])  # ["response"]
    return {"questions": questions}


@app.get("/questions-from-standard-baseline-rag")
async def get_questions_from_standard_baseline_rag():
    return await generate_questions("http://localhost:8000/query-baseline-rag")


@app.get("/questions-from-standard-graph-rag-free-form")
async def get_questions_from_standard_graph_rag_free_form():
    return await generate_questions("http://localhost:8000/query-free-form")


@app.get("/questions-from-standard-graph-rag-schema-guided")
async def get_questions_from_standard_graph_rag_schema_guided():
    return await generate_questions("http://localhost:8000/query-schema-guided")


# @app.get("/questions-from-standard-baseline_rag")
# async def get_questions_from_standard_baseline_rag():
#     prompt = """As an expert in environmental compliance, you are tasked
#             with ensuring that a project adheres to all relevant environmental
#             regulations and guidelines. Below is a summary of the borrower's
#             responsibilities related to environmental compliance. Based on this
#             information, generate a comprehensive list of questions that you
#             would ask to validate the project's compliance. Your questions
#             should cover all aspects mentioned in the summary. Ensure that
#             the questions are thorough and specific enough to identify any
#             potential compliance issues.
#             Treat every question generated as a separate entity."""

#     # response = query_index(Query(query=prompt), storage_dir="./free_form_storage")
#     # Define the URL of your FastAPI endpoint
#     url = "http://localhost:8000/query-baseline-rag"

#     import json
#     import requests

#     body = {"query": prompt, "num_retrieved_docs": 5}

#     # Send the POST request
#     response = requests.post(
#         url,
#         # files=files,  # Send the file
#         json=body,  # Send the JSON schema
#     )

#     # Print the response from the server
#     print(response.status_code)
#     print(response.json())
#     response_json = response.json()
#     # response =
#     # Now we need to parse the response and extract the questions
#     # from the generated text
#     logger.info(response)

#     response_text = response_json["response"]
#     questions = response_text.split("\n")

#     # Discard empty strings and strings with less than 5 characters
#     questions = [q for q in questions if len(q) > 5]

#     # Discard strings that doesn't contain a question mark
#     questions = [q for q in questions if "?" in q]

#     return {"questions": questions}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=6000)

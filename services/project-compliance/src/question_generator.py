import requests
from .config import config

def get_questions(prompt):
    response = requests.post(f"{config.question_generator_url}/query", json={"query": prompt})
    response.raise_for_status()
    return response.text.split("\n")

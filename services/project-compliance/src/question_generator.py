import requests
from .config import config

def get_questions(prompt):
    response = requests.post(f"{config.QUESTION_GENERATOR_URL}/generate-questions", json={"prompt": prompt})
    response.raise_for_status()
    return response.text.split("\n")

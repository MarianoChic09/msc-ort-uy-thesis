
run-graph-gen:
	cd services/graph-generation && poetry run uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload

run-question-gen:
	cd services/question-generator && poetry run uvicorn src.main:app --host 0.0.0.0 --port 6000 --reload

run-project-compliance:
	cd services/project-compliance && poetry run uvicorn src.main:app --host 0.0.0.0 --port 8540 --reload
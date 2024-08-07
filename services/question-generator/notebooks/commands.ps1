$project = "question-generator"

# -------------------------- Run locally ----------------------------
cd services/$project

micromamba create -n $project -y -c conda-forge python=3.10 

micromamba activate $project

pip install ipykernel

pip install -r requirements.txt

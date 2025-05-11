eval "$(conda shell.bash hook)"

conda activate e-library-agent
cd /app/
uvicorn app:app --host 0.0.0.0 --port 8000
conda deactivate

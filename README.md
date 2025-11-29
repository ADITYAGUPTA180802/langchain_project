# LangChain Tagline Demo (Ollama/OpenAI)

## Local run
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
uvicorn api:app --host 127.0.0.1 --port 8000
python -m http.server 5500
# Open http://localhost:5500/index.html

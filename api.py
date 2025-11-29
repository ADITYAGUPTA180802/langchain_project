import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

# ---- config via env (safer + flexible) ----
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
MODEL_NAME = os.getenv("OPENAI_MODEL", "gpt-4o-mini")  # allow override

if not OPENAI_KEY:
    raise RuntimeError("Set OPENAI_API_KEY env var on Render")

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten for production
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve your static front-end (index.html) from the repo root
app.mount("/static", StaticFiles(directory="."), name="static")

class AskIn(BaseModel):
    topic: str

prompt = ChatPromptTemplate.from_template(
    "Write one-line explanation (<=12 words) about: {topic}"
)
llm = ChatOpenAI(model=MODEL_NAME, temperature=0.4, api_key=OPENAI_KEY)
chain = prompt | llm | StrOutputParser()

@app.get("/health")
def health():
    return {"ok": True, "service": "langchain-openai", "model": MODEL_NAME}

# Open your front-end at /
@app.get("/")
def home():
    # make sure index.html is in the repo root
    if not os.path.exists("index.html"):
        return {"ok": True, "service": "langchain-openai"}
    return FileResponse("index.html")

@app.post("/ask")
def ask(body: AskIn):
    topic = (body.topic or "").strip()
    if not topic:
        raise HTTPException(status_code=400, detail="topic is required")
    try:
        text = chain.invoke({"topic": topic})
        return {"ok": True, "text": text}
    except Exception as e:
        # surface useful error text to the client
        raise HTTPException(status_code=500, detail=str(e))

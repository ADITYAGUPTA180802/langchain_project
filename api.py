import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

OPENAI_KEY = os.getenv("OPENAI_API_KEY")
MODEL_NAME = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
if not OPENAI_KEY:
    raise RuntimeError("Set OPENAI_API_KEY env var on Render")

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# serve static files (so images/css would work if you add them later)
app.mount("/static", StaticFiles(directory="."), name="static")

class AskIn(BaseModel):
    topic: str

prompt = ChatPromptTemplate.from_template("Write one-line explanation (<=12 words) about: {topic}")
llm = ChatOpenAI(model=MODEL_NAME, temperature=0.4, api_key=OPENAI_KEY)
chain = prompt | llm | StrOutputParser()

# >>> Serve your page at the ROOT URL <<<
@app.get("/")
def home():
    return FileResponse("index.html")  # index.html must be in repo root

@app.post("/ask")
def ask(body: AskIn):
return {"ok": True, "text": chain.invoke({"topic": body.topic.strip()})}


import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

OPENAI_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_KEY:
    raise RuntimeError("Set OPENAI_API_KEY env var on Render")

app = FastAPI()
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

class AskIn(BaseModel):
    topic: str

prompt = ChatPromptTemplate.from_template(
    "Write one-line explanation (<=12 words) about: {topic}"
)
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.4, api_key=OPENAI_KEY)
chain = prompt | llm | StrOutputParser()

@app.get("/")
def ping():
    return {"ok": True, "service": "langchain-openai"}

@app.post("/ask")
def ask(body: AskIn):
    return {"ok": True, "text": chain.invoke({"topic": body.topic.strip()})}
